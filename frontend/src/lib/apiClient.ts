// Base URL for all REST requests. VITE_API_URL defaults to '' (relative paths),
// which lets Nginx proxy /api/v1/* to the backend in production.
const API_BASE = (import.meta.env.VITE_API_URL as string) ?? '';
const API_KEY = (import.meta.env.VITE_API_KEY as string) ?? '';

export const API_V1 = `${API_BASE}/api/v1`;

function authHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  if (API_KEY) h['X-API-Key'] = API_KEY;
  return h;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_V1}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_V1}${path}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (!res.ok) {
    throw new Error(res.statusText);
  }
}
