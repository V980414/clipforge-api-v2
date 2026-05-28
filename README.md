# ClipForge API — Seu servidor próprio de cortes

API REST que substitui LumiClip/Vizard. Baixa do YouTube, transcreve, escolhe os melhores momentos com IA, corta em 9:16 com legendas e devolve MP4.

## Stack

- **FastAPI** (Python) — servidor HTTP rápido
- **yt-dlp** — baixa vídeo do YouTube (grátis)
- **faster-whisper** — transcrição local (grátis, roda em CPU)
- **Lovable AI (Gemini Flash)** — escolhe os cortes virais (você já tem a key)
- **FFmpeg** — corta, redimensiona pra 9:16, queima legendas

## Custo estimado

- Railway: **$5/mês** (plano Hobby, suficiente pra ~500min de vídeo/mês)
- Lovable AI: ~$0.001 por vídeo (escolha de cortes via Gemini Flash)
- **Total: R$30/mês fixo, processa praticamente ilimitado**

Compare com Vizard ($67/mês = 2000min): mesmo volume custaria 13x mais.

---

## Deploy em 5 minutos (Railway)

### 1. Criar conta no Railway

Acesse https://railway.app e faça login com GitHub.

### 2. Subir esses arquivos pro GitHub

```bash
# No seu computador, dentro desta pasta:
git init
git add .
git commit -m "ClipForge API"
git branch -M main
# Crie um repo novo em github.com/new (privado) e:
git remote add origin https://github.com/SEU_USER/clipforge-api.git
git push -u origin main
```

### 3. Deploy no Railway

1. Railway → **New Project** → **Deploy from GitHub repo** → escolha `clipforge-api`
2. Railway detecta o Dockerfile automaticamente
3. Em **Variables**, adicione:
   - `LOVABLE_API_KEY` = (mesma chave do seu projeto Lovable)
   - `API_SECRET` = uma string aleatória forte (ex: rode `openssl rand -hex 32`)
   - `WEBHOOK_URL` = (deixa em branco por enquanto, preencho depois)
4. Em **Settings** → **Networking** → **Generate Domain**. Vai gerar algo como `clipforge-api-production.up.railway.app`

### 4. Me passar de volta

- A URL do Railway (ex: `https://clipforge-api-production.up.railway.app`)
- O `API_SECRET` que você gerou

Eu uso esses dois pra conectar o Lovable ao seu servidor.

---

## API Contract

### `POST /jobs` — criar job de processamento

```json
Headers: { "x-api-secret": "<API_SECRET>" }
Body: {
  "youtube_url": "https://youtube.com/watch?v=...",
  "callback_url": "https://seu-app.lovable.app/api/public/clipforge-webhook",
  "callback_secret": "...",
  "job_id": "uuid-do-video-no-supabase"
}
Response: { "job_id": "...", "status": "queued" }
```

### `GET /jobs/{job_id}` — status

```json
Response: {
  "job_id": "...",
  "status": "queued|downloading|transcribing|analyzing|cutting|completed|failed",
  "progress": 0-100,
  "clips": [...]   // só quando completed
}
```

### Webhook (servidor chama seu Lovable)

Quando termina, chama `callback_url` com:

```json
{
  "job_id": "...",
  "status": "completed",
  "video_title": "...",
  "duration_seconds": 1800,
  "clips": [
    {
      "title": "Momento viral",
      "start_seconds": 120, "end_seconds": 180,
      "viral_score": 92, "hashtags": ["#shorts"],
      "caption": "Texto pro Instagram",
      "download_url": "https://railway.../clips/abc.mp4"
    }
  ]
}
```

---

## Estrutura

```
app/
  main.py         # FastAPI + endpoints
  pipeline.py     # download → transcribe → analyze → cut
  storage.py      # serve MP4 finalizados
Dockerfile        # imagem com ffmpeg + python
requirements.txt
```


## Correção Railway v2

Se o Railway mostrar **Build failed** ou **Deployment failed** mesmo com logs tipo `Uvicorn running`, use esta versão v2.

Mudanças aplicadas:
- Remove `ENV PORT=8000` para não brigar com a porta dinâmica do Railway.
- Usa `--port ${PORT:-8000}` com fallback local.
- Remove healthcheck obrigatório do Railway para evitar falso erro de deploy.
- Troca o modelo padrão para `base`, mais leve para começar no plano barato. Depois você pode mudar para `small` se quiser mais qualidade.

### Variáveis obrigatórias no Railway

Configure em **Variables**:

```
API_SECRET=crie_uma_senha_grande_aqui
LOVABLE_API_KEY=sua_chave_lovable
PUBLIC_BASE_URL=https://sua-url-do-railway.up.railway.app
WHISPER_MODEL=base
```

### Teste rápido

Depois do deploy ficar verde, abra:

```
https://sua-url-do-railway.up.railway.app/
```

Deve aparecer algo como:

```json
{ "service": "ClipForge API", "docs": "/docs" }
```
