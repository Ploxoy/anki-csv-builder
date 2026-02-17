import { NoticeMessage } from "../lib/messages";

type NoticeProps = {
  notice: NoticeMessage | null;
  className?: string;
};

export function Notice({ notice, className = "" }: NoticeProps) {
  if (!notice) return null;
  return (
    <div className={`notice notice-${notice.level} ${className}`.trim()}>
      <div>{notice.message}</div>
      {notice.details && (
        <details className="notice-details">
          <summary>Details</summary>
          <pre>{notice.details}</pre>
        </details>
      )}
    </div>
  );
}
