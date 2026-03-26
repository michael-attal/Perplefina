/**
 * Simple bearer token authentication for API routes.
 * Set PERPLEFINA_API_TOKEN in your environment to enable.
 * If not set, all requests are allowed (local dev mode).
 */
export function verifyApiToken(req: Request): { authorized: boolean; response?: Response } {
  const token = process.env.PERPLEFINA_API_TOKEN;

  // If no token is configured, skip auth (local dev mode)
  if (!token) {
    return { authorized: true };
  }

  const authHeader = req.headers.get('authorization');
  const bearerToken = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  if (bearerToken !== token) {
    return {
      authorized: false,
      response: Response.json(
        { message: 'Unauthorized: invalid or missing Bearer token' },
        { status: 401 },
      ),
    };
  }

  return { authorized: true };
}
