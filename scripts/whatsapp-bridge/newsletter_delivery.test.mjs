import assert from 'node:assert/strict';
import test from 'node:test';

import { deliverNewsletterMedia } from './newsletter_delivery.js';

test('newsletter delivery uses the outbound transport with the subscribed destination', async () => {
  const sent = [];
  const tracked = [];
  const result = await deliverNewsletterMedia({
    delivery: {
      destinationChatId: '120363000000000000@g.us',
      filePath: '/sanitized/goal.mp4',
      mediaKind: 'video',
      caption: 'Lions score!',
    },
    readFile: (filePath) => {
      assert.equal(filePath, '/sanitized/goal.mp4');
      return Buffer.from('sanitized-media');
    },
    send: async (chatId, payload) => {
      sent.push({ chatId, payload });
      return { key: { id: 'outbound-media-1' } };
    },
    trackSentMessageId: (message) => tracked.push(message.key.id),
  });

  assert.equal(sent.length, 1);
  assert.equal(sent[0].chatId, '120363000000000000@g.us');
  assert.deepEqual(sent[0].payload, {
    video: Buffer.from('sanitized-media'),
    caption: 'Lions score!',
    mimetype: 'video/mp4',
  });
  assert.deepEqual(tracked, ['outbound-media-1']);
  assert.equal(result.messageId, 'outbound-media-1');
});

test('GIF video media preserves WhatsApp gifPlayback on outbound transport', async () => {
  let payload;
  await deliverNewsletterMedia({
    delivery: {
      destinationChatId: 'authorized@s.whatsapp.net',
      filePath: '/sanitized/replay.mp4',
      mediaKind: 'gif',
      caption: 'Replay',
    },
    readFile: () => Buffer.from('gif-video'),
    send: async (_chatId, value) => { payload = value; return { key: { id: 'outbound-gif-1' } }; },
  });
  assert.deepEqual(payload, {
    video: Buffer.from('gif-video'),
    caption: 'Replay',
    mimetype: 'video/mp4',
    gifPlayback: true,
  });
});

test('GIF image media is not mislabeled as an MP4 video', async () => {
  let payload;
  await deliverNewsletterMedia({
    delivery: {
      destinationChatId: 'authorized@s.whatsapp.net',
      filePath: '/sanitized/replay.gif',
      mediaKind: 'gif',
      caption: 'Replay',
    },
    readFile: () => Buffer.from('gif-image'),
    send: async (_chatId, value) => { payload = value; return { key: { id: 'outbound-gif-image-1' } }; },
  });
  assert.deepEqual(payload, {
    image: Buffer.from('gif-image'),
    caption: 'Replay',
    mimetype: 'image/gif',
  });
});
