import path from 'path';

const VIDEO_MIMES = {
  mp4: 'video/mp4',
  mov: 'video/quicktime',
  avi: 'video/x-msvideo',
  mkv: 'video/x-matroska',
  '3gp': 'video/3gpp',
};

/**
 * Build and submit the one native WhatsApp media message for an already
 * matched Channel post. The caller injects the live socket transport so this
 * module remains testable without connecting to WhatsApp.
 */
export async function deliverNewsletterMedia({
  delivery,
  readFile,
  send,
  trackSentMessageId = () => {},
}) {
  if (!delivery?.destinationChatId || !delivery?.filePath) {
    throw new Error('newsletter delivery requires a destination and media file');
  }
  if (typeof readFile !== 'function' || typeof send !== 'function') {
    throw new TypeError('newsletter delivery requires readFile and send functions');
  }

  const extension = path.extname(delivery.filePath).slice(1).toLowerCase();
  const buffer = readFile(delivery.filePath);
  const payload = extension === 'gif'
    ? {
        image: buffer,
        caption: delivery.caption || undefined,
        mimetype: 'image/gif',
      }
    : {
        video: buffer,
        caption: delivery.caption || undefined,
        mimetype: VIDEO_MIMES[extension] || 'video/mp4',
        ...(delivery.mediaKind === 'gif' ? { gifPlayback: true } : {}),
      };
  const sent = await send(delivery.destinationChatId, payload);
  trackSentMessageId(sent);
  return { messageId: sent?.key?.id || '', sent };
}
