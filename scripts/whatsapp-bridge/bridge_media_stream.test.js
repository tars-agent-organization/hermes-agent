import assert from 'node:assert/strict';
import http from 'node:http';
import test from 'node:test';

import { downloadEncryptedContent } from './node_modules/@whiskeysockets/baileys/lib/Utils/messages-media.js';
import { downloadMediaWithRetry, extractBridgeEvent } from './bridge_helpers.js';

test('media socket abort rejects the download stream instead of crashing the bridge', async (t) => {
  const server = http.createServer((_req, res) => {
    res.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': '1048576',
    });
    res.write(Buffer.alloc(32));
    setImmediate(() => res.socket?.destroy());
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));

  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/media`;
  const stream = await downloadEncryptedContent(
    url,
    { cipherKey: Buffer.alloc(32), iv: Buffer.alloc(16) },
  );

  await assert.rejects(async () => {
    for await (const _chunk of stream) {
      // Consume until the deliberately aborted socket reaches the stream.
    }
  });
});

test('transient media download failures are retried before giving up', async () => {
  let attempts = 0;
  const expected = Buffer.from('video-bytes');

  const result = await downloadMediaWithRetry(
    async () => {
      attempts += 1;
      if (attempts < 3) throw new TypeError('terminated');
      return expected;
    },
    { attempts: 3, delayMs: 0 },
  );

  assert.equal(attempts, 3);
  assert.equal(result, expected);
});

test('video event retries the bridge download and preserves the attachment', async () => {
  let attempts = 0;
  const expected = Buffer.from('video-bytes');
  const msg = {
    key: { id: 'video-1' },
    message: {
      videoMessage: { mimetype: 'video/mp4' },
    },
    messageTimestamp: 1,
  };

  const event = await extractBridgeEvent({
    msg,
    chatId: 'chat@g.us',
    senderId: 'sender@lid',
    senderNumber: 'sender',
    isGroup: true,
    cacheDirs: { document: '/tmp' },
    downloadMedia: async () => {
      attempts += 1;
      if (attempts < 3) throw new TypeError('terminated');
      return expected;
    },
    writeMediaFile: async ({ buffer }) => {
      assert.equal(buffer, expected);
      return '/tmp/vid_test.mp4';
    },
  });

  assert.equal(attempts, 3);
  assert.deepEqual(event.mediaUrls, ['/tmp/vid_test.mp4']);
  assert.equal(event.body, '[video received]');
});
