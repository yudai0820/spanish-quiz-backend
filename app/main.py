import json
import os
import random

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI, OpenAIError

# .envファイルから環境変数を読み込み
load_dotenv()

# GPT用のOpenAIクライアントの初期化
gpt_client = AzureOpenAI(
    azure_endpoint=os.getenv("GPT_API_BASE"),
    api_key=os.getenv("GPT_API_KEY"),
    api_version="2024-05-01-preview",
)

# DALL-E用のOpenAIクライアントの初期化
dalle_client = AzureOpenAI(
    azure_endpoint=os.getenv("DALLE_API_BASE"),
    api_key=os.getenv("DALLE_API_KEY"),
    api_version="2024-02-01",
)

# FastAPIアプリケーションの作成
app = FastAPI()

# CORS 設定を環境変数から取得
allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/generate-quiz")
async def generate_quiz():
    try:
        # STEP 1: GPTで100個の実用的なスペイン語名詞を取得（JSON形式）
        list_prompt = (
            "日常会話で使われる実用的なスペイン語の名詞を50個ランダムに配列で生成してください。"
            '形式: ["casa", "libro", "escuela", ...]'
            "その他の文言は出力しないでください。"
        )

        list_response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはスペイン語の先生です。"},
                {"role": "user", "content": list_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # JSONパース
        nouns_json = list_response.choices[0].message.content.strip()
        nouns_list = json.loads(nouns_json)

        # STEP 2: Pythonでランダムに4つ選び、1つを正解とする
        selected_nouns = random.sample(nouns_list, 4)
        correct_noun = random.choice(selected_nouns)

        # STEP 3: DALL·Eを使用して、正解の名詞に関連する画像を生成する
        image_prompt = (
            f"A clean, simple cartoon-style illustration of the Spanish noun '{correct_noun}' "
            "suitable for a children's language-learning quiz app. "
            "The image must not contain any text, letters, numbers, symbols, or captions. "
            "Avoid signs, logos, or any written content. Focus only on visual representation of the object or concept."
        )

        image_response = dalle_client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="1024x1024",
        )
        image_url = image_response.data[0].url

        # 意味を取得する追加プロンプト
        meaning_prompt = f"スペイン語の名詞「{correct_noun}」の日本語の意味を1語で教えてください。出力はその単語のみ。"

        meaning_response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは日本語が堪能なスペイン語教師です。"},
                {"role": "user", "content": meaning_prompt},
            ],
            max_tokens=10,
            temperature=0.5,
        )

        correct_meaning = meaning_response.choices[0].message.content.strip()

        # STEP 4: 結果を返却
        return {
            "quiz_options": selected_nouns,
            "correct_answer": correct_noun,
            "correct_meaning": correct_meaning,
            "image_url": image_url,
        }

    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# もし開発環境で直接実行する場合
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
