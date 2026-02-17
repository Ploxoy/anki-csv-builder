export type NoticeLevel = "info" | "success" | "warning" | "error";

export type NoticeMessage = {
  level: NoticeLevel;
  message: string;
  details?: string | null;
};

export type GenerateNoticeKey = "input" | "run" | "audio" | "export";
export type SettingsNoticeKey = "toolbar" | "access" | "generation" | "audio" | "export";
export type AdminNoticeKey = "toolbar" | "invite" | "users" | "usage";

export type ScopedNotices = {
  generate: Record<GenerateNoticeKey, NoticeMessage | null>;
  settings: Record<SettingsNoticeKey, NoticeMessage | null>;
  admin: Record<AdminNoticeKey, NoticeMessage | null>;
};

export const EMPTY_NOTICES: ScopedNotices = {
  generate: { input: null, run: null, audio: null, export: null },
  settings: { toolbar: null, access: null, generation: null, audio: null, export: null },
  admin: { toolbar: null, invite: null, users: null, usage: null },
};

export function withGenerateNotice(
  state: ScopedNotices,
  section: GenerateNoticeKey,
  notice: NoticeMessage | null
): ScopedNotices {
  return { ...state, generate: { ...state.generate, [section]: notice } };
}

export function withSettingsNotice(
  state: ScopedNotices,
  section: SettingsNoticeKey,
  notice: NoticeMessage | null
): ScopedNotices {
  return { ...state, settings: { ...state.settings, [section]: notice } };
}

export function withAdminNotice(
  state: ScopedNotices,
  section: AdminNoticeKey,
  notice: NoticeMessage | null
): ScopedNotices {
  return { ...state, admin: { ...state.admin, [section]: notice } };
}

export function clearGenerateNotices(state: ScopedNotices): ScopedNotices {
  return { ...state, generate: { ...EMPTY_NOTICES.generate } };
}

export function clearSettingsNotices(state: ScopedNotices): ScopedNotices {
  return { ...state, settings: { ...EMPTY_NOTICES.settings } };
}

export function clearAdminNotices(state: ScopedNotices): ScopedNotices {
  return { ...state, admin: { ...EMPTY_NOTICES.admin } };
}
