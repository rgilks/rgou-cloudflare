export async function getGitCommitHash(): Promise<string> {
  if (typeof window === 'undefined' && typeof process !== 'undefined') {
    if (process.env.GITHUB_SHA) {
      return process.env.GITHUB_SHA;
    }
    try {
      const { execSync } = await import('child_process');
      return execSync('git rev-parse HEAD').toString().trim();
    } catch {
      return 'unknown';
    }
  }
  return 'unknown';
}
