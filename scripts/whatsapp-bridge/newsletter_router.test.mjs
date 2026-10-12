import { strict as assert } from 'node:assert';
import test from 'node:test';
import {
  createNewsletterRouter,
  processNewsletterUpsert,
} from './newsletter_router.js';

const NOW = 1_700_000_000_000;
const CHANNEL = 'sports@newsletter';
const DM = 'authorized@s.whatsapp.net';
const GROUP = '120363000000000000@g.us';

function makeClock(initial = NOW) {
  let value = initial;
  return { now: () => value, advance: (ms) => { value += ms; } };
}

function router(overrides = {}) {
  return createNewsletterRouter({
    allowedChannelIds: [CHANNEL],
    allowedDestinationIds: [DM, GROUP],
    now: () => NOW,
    ...overrides,
  });
}

function subscription(overrides = {}) {
  return {
    gameId: 'game-42',
    channelId: CHANNEL,
    destinationChatId: DM,
    aliases: ['Lions', 'Tigers'],
    startsAt: NOW - 60_000,
    expiresAt: NOW + 60_000,
    ...overrides,
  };
}

function videoPost(overrides = {}) {
  return {
    key: { id: 'channel-post-1', remoteJid: CHANNEL },
    messageTimestamp: Math.floor(NOW / 1000),
    message: { videoMessage: { caption: 'Lions score!', mimetype: 'video/mp4' } },
    ...overrides,
  };
}

function extracted() {
  return {
    hasMedia: true,
    mediaType: 'video',
    mediaUrls: ['/sanitized/goal.mp4'],
    body: 'Lions score!',
  };
}

test('no active subscription is discarded before extraction or delivery', async () => {
  let extractedCalls = 0;
  let delivered = 0;
  const result = await processNewsletterUpsert({
    type: 'notify', messages: [videoPost()], router: router(),
    extractEvent: async () => { extractedCalls += 1; return extracted(); },
    deliverMedia: async () => { delivered += 1; },
  });
  assert.deepEqual(result, []);
  assert.equal(extractedCalls, 0);
  assert.equal(delivered, 0);
});

test('uses the trusted origin conversation when no destination is requested', async () => {
  const subscriptions = router();
  subscriptions.register(subscription({ destinationChatId: undefined, originChatId: GROUP }));
  const deliveries = [];
  await processNewsletterUpsert({
    type: 'notify', messages: [videoPost()], router: subscriptions,
    extractEvent: async () => extracted(),
    deliverMedia: async (delivery) => deliveries.push(delivery),
  });
  assert.equal(deliveries.length, 1);
  assert.equal(deliveries[0].destinationChatId, GROUP);
});

test('delivers to separately authorized DM and group destinations', async () => {
  for (const destinationChatId of [DM, GROUP]) {
    const subscriptions = router();
    subscriptions.register(subscription({ destinationChatId }));
    const deliveries = [];
    await processNewsletterUpsert({
      type: 'notify', messages: [videoPost({ key: { id: `post-${destinationChatId}`, remoteJid: CHANNEL } })], router: subscriptions,
      extractEvent: async () => extracted(),
      deliverMedia: async (delivery) => deliveries.push(delivery),
    });
    assert.equal(deliveries.length, 1);
    assert.equal(deliveries[0].destinationChatId, destinationChatId);
    assert.equal(deliveries[0].filePath, '/sanitized/goal.mp4');
  }
});

test('rejects a destination outside the explicit allowlist without guessing a JID', () => {
  const subscriptions = router();
  assert.throws(
    () => subscriptions.register(subscription({ destinationChatId: 'some display name' })),
    /destination is not authorised/,
  );
});

test('ambiguous matches are discarded before extraction and delivery', async () => {
  const subscriptions = router();
  subscriptions.register(subscription({ gameId: 'game-a', aliases: ['Lions'] }));
  subscriptions.register(subscription({ gameId: 'game-b', aliases: ['Lions'], destinationChatId: GROUP }));
  let extractedCalls = 0;
  const result = await processNewsletterUpsert({
    type: 'notify', messages: [videoPost()], router: subscriptions,
    extractEvent: async () => { extractedCalls += 1; return extracted(); },
    deliverMedia: async () => assert.fail('ambiguous posts must not be delivered'),
  });
  assert.deepEqual(result, []);
  assert.equal(extractedCalls, 0);
});

test('deduplicates only after the real outbound transport succeeds', async () => {
  const subscriptions = router();
  subscriptions.register(subscription());
  let extractedCalls = 0;
  let delivered = 0;
  const args = {
    type: 'notify', messages: [videoPost()], router: subscriptions,
    extractEvent: async () => { extractedCalls += 1; return extracted(); },
    deliverMedia: async () => { delivered += 1; return { messageId: 'outbound-1' }; },
  };
  assert.equal((await processNewsletterUpsert(args)).length, 1);
  assert.equal((await processNewsletterUpsert(args)).length, 0);
  assert.equal(extractedCalls, 1);
  assert.equal(delivered, 1);
});

test('a failed outbound transport does not poison dedupe', async () => {
  const subscriptions = router();
  subscriptions.register(subscription());
  let attempts = 0;
  const args = {
    type: 'notify', messages: [videoPost()], router: subscriptions,
    extractEvent: async () => extracted(),
    deliverMedia: async () => {
      attempts += 1;
      if (attempts === 1) throw new Error('send failed');
      return { messageId: 'outbound-2' };
    },
  };
  assert.deepEqual(await processNewsletterUpsert(args), []);
  assert.equal((await processNewsletterUpsert(args)).length, 1);
  assert.equal(attempts, 2);
});

test('expiry removes subscriptions and prevents extraction before persistence', async () => {
  const clock = makeClock();
  const subscriptions = router({ now: clock.now });
  subscriptions.register(subscription({ expiresAt: NOW + 1 }));
  clock.advance(1);
  let extractedCalls = 0;
  const result = await processNewsletterUpsert({
    type: 'notify', messages: [videoPost()], router: subscriptions,
    extractEvent: async () => { extractedCalls += 1; return extracted(); },
    deliverMedia: async () => assert.fail('expired subscription must not deliver'),
  });
  assert.deepEqual(result, []);
  assert.equal(extractedCalls, 0);
});

test('multiple games and destinations coexist when each post has one safe match', async () => {
  const subscriptions = router();
  subscriptions.register(subscription({ gameId: 'lions-v-tigers', aliases: ['Lions'], destinationChatId: DM }));
  subscriptions.register(subscription({ gameId: 'bears-v-wolves', aliases: ['Bears'], destinationChatId: GROUP }));
  const deliveries = [];
  await processNewsletterUpsert({
    type: 'notify',
    messages: [
      videoPost({ key: { id: 'lions-1', remoteJid: CHANNEL } }),
      videoPost({ key: { id: 'bears-1', remoteJid: CHANNEL }, message: { videoMessage: { caption: 'Bears score!', mimetype: 'video/mp4' } } }),
    ],
    router: subscriptions,
    extractEvent: async () => extracted(),
    deliverMedia: async (delivery) => deliveries.push(delivery),
  });
  assert.deepEqual(deliveries.map((item) => item.destinationChatId), [DM, GROUP]);
});
