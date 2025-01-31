'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useState } from 'react';
import type { Question, Answer } from '@/types';

export default function BlockPage() {
  const { id } = useParams();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [userAnswers, setUserAnswers] = useState<{[key: string]: string}>({});
  const [showAnswers, setShowAnswers] = useState(false);

  // TODO: 実際のデータを読み込む処理を実装
  // const loadQuestions = async () => {
  //   const response = await fetch(`/api/questions/${id}`);
  //   const data = await response.json();
  //   setQuestions(data);
  // };

  const handleAnswerSelect = (questionId: string, answer: string) => {
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <Link href="/" className="text-blue-600 hover:underline">
            ← トップに戻る
          </Link>
          <h1 className="text-3xl font-bold">ブロック {typeof id === 'string' ? id.toUpperCase() : id}</h1>
          <button
            onClick={() => setShowAnswers(!showAnswers)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            {showAnswers ? '解答を隠す' : '解答を表示'}
          </button>
        </div>

        <div className="space-y-8">
          {questions.map((question) => (
            <div key={question.id} className="bg-white p-6 rounded-lg shadow">
              <div className="prose max-w-none">
                <h2 className="text-xl font-semibold mb-4">問題 {question.id}</h2>
                <p className="whitespace-pre-wrap mb-4">{question.case_text}</p>
                
                {question.sub_questions.map((subQ) => (
                  <div key={subQ.number} className="mb-6">
                    <h3 className="font-semibold mb-2">問{subQ.number}</h3>
                    <p className="mb-4">{subQ.text}</p>
                    
                    <div className="space-y-2">
                      {subQ.options.map((option, idx) => (
                        <label
                          key={idx}
                          className="flex items-center space-x-2 p-2 rounded hover:bg-gray-50"
                        >
                          <input
                            type="radio"
                            name={`${question.id}-${subQ.number}`}
                            value={String.fromCharCode(97 + idx)} // a, b, c, d, e
                            onChange={(e) => handleAnswerSelect(`${question.id}-${subQ.number}`, e.target.value)}
                            checked={userAnswers[`${question.id}-${subQ.number}`] === String.fromCharCode(97 + idx)}
                          />
                          <span>{option}</span>
                        </label>
                      ))}
                    </div>

                    {showAnswers && (
                      <div className="mt-4 p-4 bg-blue-50 rounded">
                        <h4 className="font-semibold mb-2">GPT-4oの解答</h4>
                        {/* TODO: 実際の解答データを表示 */}
                        <p>解答: a</p>
                        <p>確信度: 90%</p>
                        <p className="mt-2">解説: ...</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 