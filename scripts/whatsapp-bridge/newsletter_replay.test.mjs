import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.join(directory, 'fixtures', 'newsletter-replay.sanitized.json');

test('sanitized newsletter replay dry-runs the route without a WhatsApp send', () => {
  const stdout = execFileSync('node', ['newsletter_replay.mjs', '--fixture', fixture], {
    cwd: directory,
    encoding: 'utf8',
  });
  const report = JSON.parse(stdout);
  assert.equal(report.dryRun, true);
  assert.equal(report.deliveries.length, 1);
  assert.equal(report.deliveries[0].destinationChatId, 'origin@example.test');
  assert.equal(report.deliveries[0].filePath, '/sanitized/no-real-media.mp4');
  assert.equal(report.transportCalls, 1);
});
