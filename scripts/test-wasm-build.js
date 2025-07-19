#!/usr/bin/env node

import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

console.log('🔧 Testing WASM Build & Loading');
console.log('='.repeat(50));

try {
  console.log('📦 Building WASM assets...');
  execSync('npm run build:wasm-assets', { stdio: 'inherit' });
  console.log('✅ WASM build successful');

  console.log('📁 Checking WASM files...');
  const wasmDir = join(process.cwd(), 'public', 'wasm');
  const wasmFiles = ['rgou_ai_core_bg.wasm', 'rgou_ai_core.js'];

  for (const file of wasmFiles) {
    const filePath = join(wasmDir, file);
    if (!existsSync(filePath)) {
      throw new Error(`Missing WASM file: ${file}`);
    }

    const stats = readFileSync(filePath);
    console.log(`✅ ${file}: ${(stats.length / 1024).toFixed(1)}KB`);
  }

  console.log('🔒 Checking security headers...');
  const headersPath = join(process.cwd(), 'public', '_headers');
  if (existsSync(headersPath)) {
    const headers = readFileSync(headersPath, 'utf8');
    const requiredHeaders = [
      'Cross-Origin-Embedder-Policy: require-corp',
      'Cross-Origin-Opener-Policy: same-origin',
      'Cross-Origin-Resource-Policy: same-origin',
    ];

    for (const header of requiredHeaders) {
      if (!headers.includes(header)) {
        console.warn(`⚠️  Missing security header: ${header}`);
      } else {
        console.log(`✅ Security header found: ${header}`);
      }
    }
  } else {
    console.warn('⚠️  No _headers file found');
  }

  console.log('🧪 WASM build validation complete');
  console.log('✅ WASM files built and validated successfully');

  console.log('🎉 WASM build validation complete!');
  process.exit(0);
} catch (error) {
  console.error('❌ WASM build validation failed:', error.message);
  process.exit(1);
}
