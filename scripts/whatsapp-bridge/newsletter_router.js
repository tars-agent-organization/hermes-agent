import { getMessageContent } from './bridge_helpers.js';

const MAX_SUBSCRIPTION_MS = 12 * 60 * 60 * 1000;
const MAX_ALIASES = 16;
const MAX_ALIAS_LENGTH = 120;

function normalizedId(value) {
  return String(value || '').trim().toLowerCase();
}

function normalizedText(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLocaleLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

function parseTimestamp(value) {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return null;
  return timestamp < 100_000_000_000 ? timestamp * 1000 : timestamp;
}

function postTimestamp(msg, fallback) {
  return parseTimestamp(msg?.messageTimestamp) || fallback;
}

function postText(messageContent) {
  return messageContent?.videoMessage?.caption
    || messageContent?.imageMessage?.caption
    || messageContent?.extendedTextMessage?.text
    || messageContent?.conversation
    || '';
}

function supportedMedia(messageContent) {
  if (messageContent?.videoMessage) {
    return messageContent.videoMessage.gifPlayback ? 'gif' : 'video';
  }
  const mime = String(messageContent?.imageMessage?.mimetype || '').toLowerCase();
  return mime === 'image/gif' ? 'gif' : '';
}

function aliasMatches(text, aliases) {
  const haystack = ` ${normalizedText(text)} `;
  return aliases.some((alias) => haystack.includes(` ${alias} `));
}

function validateSubscription(input, { allowedChannelIds, allowedDestinationIds, now }) {
  const gameId = String(input?.gameId || '').trim();
  const channelId = normalizedId(input?.channelId);
  // The controller supplies originChatId from the authenticated inbound turn.
  // A missing requested destination deliberately means "reply in this turn",
  // never a display-name lookup or a guessed JID.
  const destinationChatId = normalizedId(input?.destinationChatId || input?.originChatId);
  const aliases = Array.isArray(input?.aliases)
    ? [...new Set(input.aliases.map(normalizedText).filter(Boolean))]
    : [];
  const startsAt = parseTimestamp(input?.startsAt);
  const expiresAt = parseTimestamp(input?.expiresAt);

  if (!gameId || gameId.length > 160) throw new Error('gameId is required');
  if (!allowedChannelIds.has(channelId)) throw new Error('channel is not allowlisted');
  if (!allowedDestinationIds.has(destinationChatId)) throw new Error('destination is not authorised');
  if (!startsAt || !expiresAt || startsAt >= expiresAt || expiresAt - startsAt > MAX_SUBSCRIPTION_MS) {
    throw new Error('subscription window is invalid');
  }
  if (expiresAt <= now()) throw new Error('subscription window has expired');
  if (!aliases.length || aliases.length > MAX_ALIASES || aliases.some((alias) => alias.length > MAX_ALIAS_LENGTH)) {
    throw new Error('at least one valid alias is required');
  }
  return { gameId, channelId, destinationChatId, aliases, startsAt, expiresAt };
}

/**
 * Ephemeral, process-local routing for a deliberately small set of active
 * match subscriptions. It intentionally has no disk store or channel catalog.
 */
export function createNewsletterRouter({ allowedChannelIds = [], allowedDestinationIds = [], now = Date.now } = {}) {
  const allowedChannels = new Set(allowedChannelIds.map(normalizedId).filter(Boolean));
  const allowedDestinations = new Set(allowedDestinationIds.map(normalizedId).filter(Boolean));
  const subscriptions = new Map();
  const relevantPostIds = new Set();
  let subscriptionSequence = 0;

  function pruneExpired() {
    const current = now();
    for (const [id, item] of subscriptions) {
      if (item.expiresAt <= current) subscriptions.delete(id);
    }
  }

  function register(input) {
    pruneExpired();
    const item = validateSubscription(input, { allowedChannelIds: allowedChannels, allowedDestinationIds: allowedDestinations, now });
    const id = `subscription-${++subscriptionSequence}`;
    subscriptions.set(id, { id, ...item });
    return { id, gameId: item.gameId, expiresAt: item.expiresAt };
  }

  function remove(id) {
    return subscriptions.delete(String(id || ''));
  }

  function inspect(msg) {
    pruneExpired();
    const channelId = normalizedId(msg?.key?.remoteJid);
    if (!allowedChannels.has(channelId)) return null;
    const content = getMessageContent(msg);
    const mediaKind = supportedMedia(content);
    if (!mediaKind) return null;
    const when = postTimestamp(msg, now());
    const text = postText(content);
    const matches = [];
    for (const item of subscriptions.values()) {
      if (item.channelId !== channelId || when < item.startsAt || when > item.expiresAt) continue;
      if (aliasMatches(text, item.aliases)) matches.push(item);
    }
    // Ambiguity is a hard stop: a Channel post must identify one subscribed game.
    if (matches.length !== 1) return null;
    return { subscription: matches[0], mediaKind, postId: String(msg?.key?.id || '') };
  }

  function hasRelevantPost(postId) {
    return relevantPostIds.has(postId);
  }

  function rememberRelevantPost(postId) {
    if (!postId) return;
    relevantPostIds.add(postId);
    while (relevantPostIds.size > 512) relevantPostIds.delete(relevantPostIds.values().next().value);
  }

  return { register, remove, inspect, hasRelevantPost, rememberRelevantPost };
}

/**
 * Applies newsletter routing before extraction. This ordering is the safety
 * boundary: unmatched posts are neither downloaded nor persisted. Successful
 * posts go to the injected outbound transport, never the inbound conversation
 * queue.
 */
export async function processNewsletterUpsert({ type, messages, router, extractEvent, deliverMedia }) {
  if (!router || (type !== 'notify' && type !== 'append')) return [];
  if (typeof extractEvent !== 'function' || typeof deliverMedia !== 'function') return [];
  const routed = [];
  for (const msg of messages || []) {
    const decision = router.inspect(msg);
    if (!decision || !decision.postId || router.hasRelevantPost(decision.postId)) continue;
    let extracted;
    try {
      extracted = await extractEvent(msg, decision);
    } catch {
      // Channel delivery is best-effort and intentionally silent on media errors.
      continue;
    }
    if (!extracted?.hasMedia || !Array.isArray(extracted.mediaUrls) || extracted.mediaUrls.length === 0) continue;

    const delivery = {
      destinationChatId: decision.subscription.destinationChatId,
      filePath: extracted.mediaUrls[0],
      mediaKind: decision.mediaKind,
      caption: extracted.body || '',
      gameId: decision.subscription.gameId,
      postId: decision.postId,
    };
    try {
      // Only remember the post once the real outbound transport resolves.
      // This leaves a transient send failure eligible for a later retry.
      const sent = await deliverMedia(delivery);
      router.rememberRelevantPost(decision.postId);
      routed.push({ ...delivery, sent });
    } catch {
      // Channel delivery is best-effort and intentionally silent on send errors.
    }
  }
  return routed;
}

export function parseNewsletterAllowlist(value) {
  return String(value || '').split(',').map(normalizedId).filter(Boolean);
}
