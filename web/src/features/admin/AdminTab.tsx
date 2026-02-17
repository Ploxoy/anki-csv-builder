import { useState } from "react";
import { InviteCreateResponse, UsageListResponse, UserListResponse } from "../../types";
import { NoticeMessage } from "../../lib/messages";
import { shortJson } from "../../lib/uiState";
import { Notice } from "../../ui/Notice";

type AdminTabProps = {
  adminBusy: boolean;
  hasAdminKey: boolean;
  users: UserListResponse | null;
  newInvite: InviteCreateResponse | null;
  adminUsage: UsageListResponse | null;
  onCreateInvite: () => void;
  onListUsers: () => void;
  onSetStatus: (userId: string, status: "active" | "blocked") => void;
  onRotate: (userId: string) => void;
  onLoadUsage: (userId: string) => void;
  notices: {
    toolbar: NoticeMessage | null;
    invite: NoticeMessage | null;
    users: NoticeMessage | null;
    usage: NoticeMessage | null;
  };
};

export function AdminTab({
  adminBusy,
  hasAdminKey,
  users,
  newInvite,
  adminUsage,
  onCreateInvite,
  onListUsers,
  onSetStatus,
  onRotate,
  onLoadUsage,
  notices,
}: AdminTabProps) {
  const [copied, setCopied] = useState(false);

  async function copyInviteToken(token: string) {
    try {
      await navigator.clipboard.writeText(token);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="tab-layout">
      <section className="card">
        <div className="section-head">
          <h2>User management</h2>
          <span className="muted">Admin only</span>
        </div>

        <div className="toolbar row-wrap">
          <div className="toolbar-actions">
            <button className="btn primary" type="button" onClick={onCreateInvite} disabled={adminBusy || !hasAdminKey}>
              Create invite
            </button>
            <button className="btn" type="button" onClick={onListUsers} disabled={adminBusy || !hasAdminKey}>
              List users
            </button>
          </div>
        </div>

        {!hasAdminKey && <div className="empty-state">Add X-API-Key in Settings to unlock admin actions.</div>}

        <Notice notice={notices.toolbar} />
        <Notice notice={notices.invite} />

        {newInvite && (
          <article className="result-row invite-card">
            <div className="result-main">
              <p>
                <strong>User id:</strong> {newInvite.user_id}
              </p>
              <p>
                <strong>Token:</strong> {newInvite.token}
              </p>
              <div className="toolbar-actions">
                <button className="btn tiny" type="button" onClick={() => copyInviteToken(newInvite.token)}>
                  {copied ? "Copied" : "Copy token"}
                </button>
                <span className="hint subtle">Show once: store it now.</span>
              </div>
            </div>
          </article>
        )}

        <Notice notice={notices.users} />

        {!users && hasAdminKey && <div className="empty-state">No users loaded yet.</div>}

        {users && (
          <div className="table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>id</th>
                  <th>status</th>
                  <th>label</th>
                  <th>created</th>
                  <th>last_used</th>
                  <th>actions</th>
                </tr>
              </thead>
              <tbody>
                {users.items.map((user) => {
                  const nextStatus = user.status === "active" ? "blocked" : "active";
                  return (
                    <tr key={user.id}>
                      <td>{user.id}</td>
                      <td>
                        <span className={`status-pill ${user.status === "active" ? "ok" : "failed"}`}>{user.status}</span>
                      </td>
                      <td>{user.label || "-"}</td>
                      <td>{user.created_at}</td>
                      <td>{user.last_used_at || "-"}</td>
                      <td>
                        <div className="toolbar-actions">
                          <button
                            className="btn tiny"
                            type="button"
                            onClick={() => onSetStatus(user.id, nextStatus)}
                            disabled={adminBusy || !hasAdminKey}
                          >
                            {nextStatus === "blocked" ? "Block" : "Unblock"}
                          </button>
                          <button
                            className="btn tiny"
                            type="button"
                            onClick={() => onRotate(user.id)}
                            disabled={adminBusy || !hasAdminKey}
                          >
                            Rotate
                          </button>
                          <button
                            className="btn tiny"
                            type="button"
                            onClick={() => onLoadUsage(user.id)}
                            disabled={adminBusy || !hasAdminKey}
                          >
                            Usage
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card">
        <div className="section-head">
          <h2>Admin usage</h2>
        </div>

        <Notice notice={notices.usage} />

        {!adminUsage && <div className="empty-state">Select a user and click Usage to load data.</div>}

        {adminUsage && (
          <>
            <div className="summary-strip">
              <div className="summary-item">
                <span className="k">user</span> {adminUsage.user_id}
              </div>
              <div className="summary-item">
                <span className="k">events</span> {adminUsage.summary.events}
              </div>
              <div className="summary-item">
                <span className="k">tokens</span> {adminUsage.summary.input_tokens}+{adminUsage.summary.output_tokens}
              </div>
              <div className="summary-item">
                <span className="k">cached</span> {adminUsage.summary.cached_tokens}
              </div>
              <div className="summary-item">
                <span className="k">audio chars</span> {adminUsage.summary.audio_chars}
              </div>
              {adminUsage.summary.raw_cost_usd != null && (
                <div className="summary-item">
                  <span className="k">est USD</span> {adminUsage.summary.raw_cost_usd}
                </div>
              )}
            </div>

            <details className="raw-block">
              <summary>Raw usage</summary>
              <pre>{shortJson(adminUsage)}</pre>
            </details>
          </>
        )}
      </section>
    </section>
  );
}
