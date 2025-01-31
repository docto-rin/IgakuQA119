import { NextResponse } from 'next/server';
import type { Question } from '@/types';

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // TODO: 実際のデータ取得処理を実装
    // 現在はモックデータを返しています
    const mockData: Question[] = [
      {
        id: '119A1',
        case_text: '54歳の男性。発熱と咳を主訴に来院した。\n3日前から38℃台の発熱と咳が出現し...',
        has_image: false,
        sub_questions: [
          {
            number: 1,
            text: '最も考えられる診断は何か。',
            options: [
              '市中肺炎',
              '気管支喘息',
              '急性気管支炎',
              '肺結核',
              '肺癌'
            ]
          }
        ]
      }
    ];

    return NextResponse.json(mockData);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch questions' },
      { status: 500 }
    );
  }
} 