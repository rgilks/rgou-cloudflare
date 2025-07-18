import { getFileHash } from './getFileHash';

export async function getClassicAIVersion(): Promise<string> {
  if (typeof window === 'undefined' && typeof process !== 'undefined') {
    try {
      const { readFileSync } = await import('fs');
      const { createHash } = await import('crypto');

      const cargoTomlPath = 'worker/rust_ai_core/Cargo.toml';
      const cargoContent = readFileSync(cargoTomlPath, 'utf8');
      const versionMatch = cargoContent.match(/version = "([^"]+)"/);
      const crateVersion = versionMatch ? versionMatch[1] : '0.1.0';
      const aiSourceFiles = [
        'worker/rust_ai_core/src/lib.rs',
        'worker/rust_ai_core/src/features.rs',
        'worker/rust_ai_core/src/ml_ai.rs',
        'worker/rust_ai_core/src/neural_network.rs',
      ];

      const hash = createHash('sha256');
      for (const file of aiSourceFiles) {
        try {
          const content = readFileSync(file, 'utf8');
          hash.update(content);
        } catch {
        }
      }

      const aiHash = hash.digest('hex').slice(0, 8);
      return `${crateVersion}-${aiHash}`;
    } catch {
      return 'unknown';
    }
  }
  return 'unknown';
}

export async function getMLAIVersion(): Promise<string> {
  return await getFileHash('public/ml-weights.json.gz');
}
