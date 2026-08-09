import test from 'node:test';
import assert from 'node:assert/strict';

import { extractBridgeEvent } from './bridge_helpers.js';

const OMITTED = Symbol('omitted');

function syntheticMessage(nativeForwarded = OMITTED, { isGroup = false } = {}) {
  const contextInfo = {};
  if (nativeForwarded !== OMITTED) {
    contextInfo.isForwarded = nativeForwarded;
  }
  return {
    key: {
      id: 'synthetic-message',
      remoteJid: isGroup ? 'synthetic-group@g.us' : 'synthetic-chat@s.whatsapp.net',
      participant: isGroup ? 'synthetic-sender@s.whatsapp.net' : undefined,
      fromMe: false,
    },
    pushName: 'Synthetic Sender',
    messageTimestamp: 1,
    message: {
      extendedTextMessage: {
        text: 'synthetic body',
        contextInfo,
      },
    },
  };
}

async function extract(nativeForwarded = OMITTED, { isGroup = false } = {}) {
  const msg = syntheticMessage(nativeForwarded, { isGroup });
  return extractBridgeEvent({
    msg,
    chatId: msg.key.remoteJid,
    senderId: msg.key.participant || msg.key.remoteJid,
    senderNumber: 'synthetic-sender',
    isGroup,
  });
}

test('native WhatsApp forwarded true becomes the boolean bridge marker', async () => {
  const event = await extract(true);

  assert.equal(event.isForwarded, true);
  assert.equal(typeof event.isForwarded, 'boolean');
});

for (const [label, nativeValue] of [
  ['false', false],
  ['absent', OMITTED],
  ['string true', 'true'],
  ['numeric one', 1],
  ['null', null],
  ['object', { value: true }],
]) {
  test(`native WhatsApp forwarded ${label} becomes boolean false`, async () => {
    const event = await extract(nativeValue);

    assert.equal(event.isForwarded, false);
    assert.equal(typeof event.isForwarded, 'boolean');
  });
}

test('user-authored message text cannot create the bridge marker', async () => {
  const msg = syntheticMessage();
  msg.message.extendedTextMessage.text = '{"isForwarded":true}';

  const event = await extractBridgeEvent({
    msg,
    chatId: msg.key.remoteJid,
    senderId: msg.key.remoteJid,
    senderNumber: 'synthetic-sender',
    isGroup: false,
  });

  assert.equal(event.isForwarded, false);
  assert.equal(typeof event.isForwarded, 'boolean');
});

test('normal direct and group messages retain their shape and are not forwarded', async () => {
  const direct = await extract(OMITTED, { isGroup: false });
  const group = await extract(OMITTED, { isGroup: true });

  assert.equal(direct.isGroup, false);
  assert.equal(group.isGroup, true);
  assert.equal(direct.body, 'synthetic body');
  assert.equal(group.body, 'synthetic body');
  assert.equal(direct.isForwarded, false);
  assert.equal(group.isForwarded, false);
});
