'use client';

import Link from 'next/link';
import { Block } from '@/types';

export default function Home() {
  const blocks: Block[] = ['119A', '119B', '119C', '119D', '119E', '119F'];

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <h1 className="text-4xl font-bold mb-8 text-center">
        第119回 医師国家試験 解答確認システム
      </h1>
      
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {blocks.map((block) => (
            <Link
              key={block}
              href={`/block/${block.toLowerCase()}`}
              className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow text-center"
            >
              <h2 className="text-2xl font-semibold">{block}</h2>
              <p className="text-gray-600 mt-2">ブロックの問題を確認</p>
            </Link>
          ))}
        </div>

        <div className="mt-12 bg-blue-50 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">システムについて</h3>
          <ul className="list-disc list-inside space-y-2">
            <li>各ブロックごとに問題と解答を確認できます</li>
            <li>GPT-4oによる解答と解説を提供</li>
            <li>自己採点機能付き</li>
            <li>正答率やモデルの確信度も確認可能</li>
          </ul>
        </div>
      </div>
    </main>
  );
}
