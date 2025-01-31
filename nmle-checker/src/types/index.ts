export type Question = {
  id: string;
  case_text: string;
  sub_questions: SubQuestion[];
  has_image: boolean;
};

export type SubQuestion = {
  number: number;
  text: string;
  options: string[];
};

export type Answer = {
  answer: string;
  confidence: number;
  explanation?: string;
};

export type ModelAnswers = {
  [questionId: string]: {
    answers: {
      [modelName: string]: Answer;
    };
  };
};

export type UserAnswer = {
  questionId: string;
  subQuestionNumber: number;
  selectedAnswer: string;
};

export type Block = '119A' | '119B' | '119C' | '119D' | '119E' | '119F'; 