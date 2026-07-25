---
name: Infisical access
description: Как получить секреты из Infisical через универсальную аутентификацию (client credentials)
---

# Доступ к Infisical

## Учётные данные
Хранятся в Replit Secrets:
- `INFISICAL_CLIENT_ID`
- `INFISICAL_CLIENT_SECRET`

## Project ID
`555e71be-4c53-4b3e-9409-0d9838aea8b6`

## Рабочий shell-скрипт (одним блоком)

```bash
TOKEN=$(curl -s -X POST "https://app.infisical.com/api/v1/auth/universal-auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"clientId\":\"$INFISICAL_CLIENT_ID\",\"clientSecret\":\"$INFISICAL_CLIENT_SECRET\"}" | \
  node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{const o=JSON.parse(d);console.log(o.accessToken);})")

curl -s "https://app.infisical.com/api/v3/secrets/raw?workspaceId=555e71be-4c53-4b3e-9409-0d9838aea8b6&environment=dev" \
  -H "Authorization: Bearer $TOKEN" | \
  node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{
    const o=JSON.parse(d);
    if(o.secrets) o.secrets.forEach(s=>console.log(s.secretKey+'='+s.secretValue));
    else console.log('Error:', JSON.stringify(o).slice(0,400));
  })"
```

**Why:**
- `process.env` недоступен в CodeExecution sandbox — только ShellExec читает переменные окружения
- `python3` не установлен в среде — использовать `node` для парсинга JSON
- Токен живёт в рамках одного shell-вызова: получаем и сразу используем
- API v3 `/secrets/raw` возвращает поле `secrets[].secretValue` в открытом виде

**How to apply:**
Запускать через `ShellExec`. Если нужны конкретные ключи — фильтровать через `.filter(s=>keys.includes(s.secretKey))` в node-пайпе.

## Найденные Supabase-реквизиты (amelitacoffey4d162)
- `SUPABASE_CP_amelitacoffey4d162` — полная строка подключения PostgreSQL
- `SUPABASE_URL_amelitacoffey4d162` — `https://tqbljevvatiqvqodqmkq.supabase.co`
- `SUPABASE_PROJECT_ID_amelitacoffey4d162` — `tqbljevvatiqvqodqmkq`
- `SUPABASE_DB_HOST_amelitacoffey4d162` — pooler: `aws-0-eu-central-1.pooler.supabase.com`
- `SUPABASE_DB_USER_amelitacoffey4d162` — `postgres.tqbljevvatiqvqodqmkq`
- `SUPABASE_DB_PASSWORD_amelitacoffey4d162` — в Infisical
- `SUPABASE_DB_PORT_amelitacoffey4d162` — `5432`
- `SUPABASE_DB_NAME_amelitacoffey4d162` — `postgres`
