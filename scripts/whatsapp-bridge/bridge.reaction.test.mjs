import { strict as assert } from 'node:assert';

import { buildReactionPayload } from './bridge_helpers.js';


{
  const payload = buildReactionPayload({
    chatId: '120363000000000000@g.us',
    messageId: 'message-1',
    participant: '15550001111@s.whatsapp.net',
    emoji: '👀',
  });

  assert.deepEqual(payload, {
    react: {
      text: '👀',
      key: {
        remoteJid: '120363000000000000@g.us',
        id: 'message-1',
        fromMe: false,
        participant: '15550001111@s.whatsapp.net',
      },
    },
  });
}

{
  const payload = buildReactionPayload({
    chatId: '120363000000000000@g.us',
    messageId: 'message-1',
    participant: '15550001111@s.whatsapp.net',
    emoji: '',
  });

  assert.equal(payload.react.text, '');
}

assert.throws(
  () => buildReactionPayload({ chatId: '', messageId: 'message-1', emoji: '👀' }),
  /chatId and messageId are required/,
);
