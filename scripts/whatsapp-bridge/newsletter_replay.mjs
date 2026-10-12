#!/usr/bin/env node
import { readFileSync } from 'fs';
import { createNewsletterRouter, processNewsletterUpsert } from './newsletter_router.js';

const args = process.argv.slice(2);
const fixtureIndex = args.indexOf('--fixture');
const fixturePath = fixtureIndex >= 0 ? args[fixtureIndex + 1] : '';

if (!fixturePath || args.includes('--send')) {
  console.error('Usage: node newsletter_replay.mjs --fixture <sanitized-fixture.json>');
  console.error('Replay is dry-run only and never connects to or sends WhatsApp media.');
  process.exitCode = 2;
} else {
  const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
  const now = Number(fixture.now);
  if (!Number.isFinite(now)) throw new Error('fixture.now must be a timestamp');
  const router = createNewsletterRouter({
    allowedChannelIds: fixture.allowedChannelIds,
    allowedDestinationIds: fixture.allowedDestinationIds,
    now: () => now,
  });
  router.register(fixture.subscription);

  const deliveries = [];
  let transportCalls = 0;
  await processNewsletterUpsert({
    type: 'notify',
    messages: fixture.posts,
    router,
    // The fixture supplies only synthetic saved-media metadata. This callback
    // never invokes Baileys or downloads any bytes.
    extractEvent: async () => fixture.extracted,
    // Dry-run transport records what would be submitted but has no socket and
    // therefore cannot send a WhatsApp message.
    deliverMedia: async (delivery) => {
      transportCalls += 1;
      deliveries.push(delivery);
      return { dryRun: true };
    },
  });
  process.stdout.write(`${JSON.stringify({ dryRun: true, transportCalls, deliveries })}\n`);
}
