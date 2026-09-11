# Mini Desafio RAG — VendeFácil Knowledge Base

Projeto desenvolvido para o Mini Desafio de Assistente RAG Corporativo da VendeFácil Tecnologia Ltda.

## Contexto

A VendeFácil fornece soluções de automação comercial para pequenos e médios varejistas. O projeto constrói um assistente RAG capaz de processar fontes internas da empresa, recuperar informações relevantes com busca híbrida e filtros por metadados, e responder perguntas com evidências citadas e proteção de dados conforme a LGPD. A avaliação final está em [`RELATORIO.md`](RELATORIO.md).

Os principais produtos da empresa são:

1. **VendeFácil PDV:** sistema de frente de caixa.
2. **VendeFácil Estoque:** gestão de inventário e entrada de NF-e.
3. **VendeFácil Loja:** e-commerce omnicanal e catálogo digital.
4. **VendeFácil Analytics:** dashboards, DRE e curva ABC.
5. **VendeFácil Pay:** pagamentos com TEF IP e PIX dinâmico.

## Etapa 1 — Ingestão, metadados e índice FAISS

Nesta etapa, implementamos:

- ingestão de dados heterogêneos;
- chunking adaptativo conforme a natureza de cada fonte;
- metadados padronizados;
- validação de `chunk_id` únicos;
- criação e persistência de índice vetorial FAISS;
- recarga do índice sem reindexar o corpus;
- script de sanidade para verificar a recuperação vetorial.

## Formatos processados

| Formato | Fontes | Estratégia de chunking |
|---|---|---|
| CSV | clientes, funcionários e vendas | Um registro por chunk, serializado em texto legível |
| JSON | produtos e lojas | Um objeto por chunk |
| JSONL | tickets de suporte | Um ticket por chunk; quando necessário, divide apenas o corpo e repete o cabeçalho |
| Markdown | manuais, documentação e atas | Divisão por cabeçalhos, com fallback por tamanho |
| PDF | políticas internas | Divisão por página, parágrafos ou cláusulas com overlap |
| TXT | e-mails | Divisão por mensagem do thread antes da divisão por tamanho |
| CSV de logs | logs de sistema | Um log por chunk |

## Estrutura do projeto

```text
rag-vendefacil-grupo07-neves-fregulha/
├── data/
│   ├── structured/
│   │   ├── customers.csv
│   │   ├── employees.csv
│   │   ├── products.json
│   │   ├── sales.csv
│   │   └── stores.json
│   ├── semi_structured/
│   │   ├── system_logs.csv
│   │   └── tickets.jsonl
│   └── unstructured/
│       ├── documentation/
│       ├── emails/
│       ├── meetings/
│       └── policies/
├── benchmark/
│   └── questions_and_ground_truth.json   ← 24 perguntas com gabarito (Etapa 4)
├── src/
│   ├── ingest.py            ← Etapa 1: ingestão
│   ├── metadata.py
│   ├── sanity_check.py
│   ├── vectorstore.py
│   ├── loaders/
│   │   ├── structured.py
│   │   ├── log_loader.py
│   │   ├── jsonl_loader.py
│   │   ├── markdown_loader.py
│   │   ├── pdf_loader.py
│   │   └── text_loader.py
│   ├── query_analyzer.py    ← Etapa 2: filtros extraídos da pergunta
│   ├── retrieve.py          ← Etapa 2: busca híbrida (FAISS + BM25, RRF)
│   ├── retrieval_check.py   ← Etapa 2: comparativo com e sem filtro
│   ├── schema.py            ← Etapa 3: modelos Pydantic
│   ├── policy.py            ← Etapa 3: política de LGPD e escopo
│   └── rag.py               ← Etapa 3: pipeline de geração estruturada
├── eval/
│   ├── run_benchmark.py     ← Etapa 4: executa o benchmark
│   ├── judge_prompt.py      ← Etapa 4: RAG Triad (LLM-as-judge)
│   ├── score.py             ← Etapa 4: nota pela rubrica do guia
│   ├── retrieval_score.py   ← Etapa 4: mede a recuperação sem LLM
│   ├── check_providers.py   ← Etapa 4: verifica cota dos provedores
│   ├── results.json
│   ├── triad_scores.json
│   └── runs/                ← histórico datado das medições
├── tests/                   ← 16 arquivos de teste, um por módulo
├── starter/
│   ├── requirements.txt
│   └── schema.py
├── app.py                   ← Etapa 4: interface Streamlit
├── config.py                ← provedor de LLM e parâmetros
├── .env.example
├── requirements.txt
├── ACOMPANHAMENTO.md
├── RELATORIO.md             ← Etapa 4: resultados e diagnóstico
├── RESULTADOS_ETAPA2.md
└── README.md
```

A pasta `index/` é gerada localmente durante a indexação e não é enviada ao Git, pois está no `.gitignore`.

## Metadados

Todo chunk possui os seguintes campos obrigatórios:

```python
{
    "source_file": "nome do arquivo de origem",
    "doc_type": "tipo do documento",
    "chunk_id": "identificador único e estável",
    "sensitivity": "publico, interno ou restrito",
}
```

Quando aplicável, também são usados campos como:

```text
customer_id, state, module, priority, status, date, section,
timestamp, level, service, event e error_code.
```

Os tipos de documento presentes no corpus são:

```text
ata, customer, email, employee, log, manual, policy,
product, sale, store e ticket
```

## Pré-requisitos da Etapa 1

- Python 3.10 ou superior;
- conexão com a internet na primeira indexação, para baixar o modelo local;
- não é necessária chave de API nesta etapa.

O projeto utiliza embeddings locais com o modelo multilíngue:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

A chave de API de um LLM será necessária nas etapas futuras, principalmente na Etapa 3, para gerar respostas estruturadas e aplicar guardrails de LGPD.

## Instalação

No PowerShell, dentro da pasta do repositório:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Executar a ingestão

O comando abaixo carrega e valida todas as fontes do corpus:

```powershell
.\venv\Scripts\python.exe -m src.ingest
```

Resultado esperado:

```text
Total de chunks: 5718
```

Distribuição esperada:

| doc_type | Quantidade |
|---|---:|
| ata | 38 |
| customer | 2000 |
| email | 43 |
| employee | 10 |
| log | 450 |
| manual | 24 |
| policy | 23 |
| product | 5 |
| sale | 3000 |
| store | 50 |
| ticket | 75 |

## Criar e persistir o índice FAISS

Para vetorizar o corpus e salvar o índice em disco:

```powershell
.\venv\Scripts\python.exe -m src.vectorstore
```

Os arquivos do índice serão criados na pasta:

```text
index/
├── index.faiss
└── index.pkl
```

## Recarregar o índice sem reindexar

Após criar o índice uma vez, ele pode ser recarregado sem processar novamente os arquivos:

```powershell
.\venv\Scripts\python.exe -c "from src.vectorstore import load_vectorstore; vectorstore = load_vectorstore(); print(f'Documentos no indice: {vectorstore.index.ntotal}')"
```

Resultado esperado:

```text
Documentos no indice: 5718
```

## Executar a sanidade do índice

O script de sanidade imprime o total de chunks, a distribuição por `doc_type` e os cinco resultados mais similares para cada pergunta:

```powershell
.\venv\Scripts\python.exe -m src.sanity_check
```

Perguntas de teste:

1. `Quais produtos a VendeFácil oferece?`
2. `Quais lojas estão localizadas em Minas Gerais?`
3. `Quais erros ocorreram no módulo de PDV?`

A recuperação híbrida e os filtros por metadados são implementados na Etapa 2.

## Etapa 2 — Busca híbrida e filtros por metadados

A recuperação em `src/retrieve.py` segue este fluxo:

1. o Query Analyzer baseado em regras extrai `doc_type`, `state`, `module`, `customer_id` e `priority`;
2. os filtros são normalizados e validados contra os valores realmente presentes no índice; um valor inválido retorna zero resultados em vez de ampliar silenciosamente a consulta;
3. o FAISS recupera um conjunto ampliado (`fetch_k=500`) antes de aplicar o filtro;
4. o BM25Plus busca termos exatos no corpus pré-filtrado;
5. os rankings denso e esparso são combinados por Reciprocal Rank Fusion (RRF), com `k=60`.

Escolhemos o Query Analyzer por regras porque o vocabulário de filtros do corpus é fechado e pequeno. A abordagem é determinística, não exige chave de API e impede que valores inventados sejam enviados ao FAISS. A contrapartida é ampliar explicitamente o dicionário de sinônimos quando surgir um novo modo de formular a pergunta.

Não somamos os scores do FAISS e do BM25, pois eles possuem escalas diferentes. O RRF usa somente as posições nos rankings:

```text
score_RRF(documento) = soma(1 / (60 + posição_no_ranking))
```

O BM25 é especialmente útil para códigos de erro, IDs e nomes exatos. A busca densa é mais adequada para paráfrases e sinônimos; a fusão preserva as duas capacidades.

### Executar o comparativo com e sem filtro

Crie o índice conforme a Etapa 1 e execute:

```powershell
.\venv\Scripts\python.exe -m src.retrieval_check
```

O script imprime resultados lado a lado para três perguntas específicas por estado e módulo:

1. `Quais tickets de Minas Gerais estao relacionados ao modulo de estoque?`
2. `Quais tickets de Sao Paulo estao relacionados ao modulo de PDV?`
3. `Quais tickets do Rio de Janeiro estao relacionados ao modulo pay?`

Cada linha mostra `chunk_id`, arquivo de origem, estado, módulo e uma prévia do conteúdo. A coluna filtrada deve conter somente documentos que satisfaçam simultaneamente os metadados extraídos.

Os resultados reais obtidos com o índice de 5.718 chunks estão documentados em [`RESULTADOS_ETAPA2.md`](RESULTADOS_ETAPA2.md).

## Testes

Para executar todos os testes automatizados:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Resultado esperado após a Etapa 2:

```text
40 passed
```

Sem a pasta local `index/`, os dois testes de persistência são ignorados até que `python -m src.vectorstore` seja executado.

## Etapa 3 — Síntese estruturada, evidências e guardrails de LGPD

### Schema estruturado (`src/schema.py`)

A resposta do assistente é validada por dois modelos Pydantic, baseados em `starter/schema.py`:

- `SourceEvidence`: exige `filepath`, `chunk_id`, `quotation` (trecho literal, até 400 caracteres) e opcionalmente `doc_type`. Campos como `doc_type` usam `Literal` em vez de `str` livre, para impedir que o LLM escreva variações do mesmo valor (ex.: "Alta" vs "ALTA").
- `RAGResponse`: reúne `answer`, `confidence_level`, `sources_used`, `reasoning`, `is_refusal` e `refusal_reason`.

Um `model_validator` garante a consistência estrutural da resposta, independentemente do prompt:

- se `is_refusal=True`, `confidence_level` deve ser `"Recusado"`, `refusal_reason` deve estar preenchido e `sources_used` deve estar vazia;
- se `is_refusal=False`, `confidence_level` não pode ser `"Recusado"`, `refusal_reason` deve ser `None` e é obrigatório citar ao menos uma evidência em `sources_used`.

Optamos por aplicar essa regra como validador Pydantic, e não apenas como instrução de prompt, porque um LLM pode ignorar instruções textuais sob pressão de outros trechos do prompt; o validador impede a inconsistência na saída independentemente do que o modelo gerar.

### Política de LGPD e escopo (`src/policy.py`)

As regras de `decide_policy()` derivam diretamente de `data/unstructured/policies/seguranca_lgpd.md` (itens 1.2 a 1.4) e resultam em três níveis:

| Nível | Quando ocorre | Resultado |
|---|---|---|
| `recusar` | Pergunta fora do domínio VendeFácil, ou pede diretamente dado protegido (salário, CPF, dados de saúde) ou credencial (senha, cartão, chave de API) | `is_refusal=True`, com `refusal_reason` em `OUT_OF_DOMAIN`, `LGPD_PROTECTION` ou `CREDENTIAL_PROTECTION`; a recusa ocorre **antes** da busca, sem consultar o índice |
| `mascarar` | Pergunta legítima, mas alguma fonte recuperada tem `sensitivity="restrito"` | O conteúdo dos chunks é mascarado (`mask_sensitive_text`) antes de ir para o contexto do LLM — CPF, cartão, CVV, e-mail, telefone e valor de salário nunca chegam ao prompt em texto puro |
| `responder` | Pergunta e fontes dentro da política | Segue o fluxo normal |

O vocabulário de domínio (`DOMAIN_KEYWORDS`) foi ampliado durante a validação: perguntas como "Qual a média salarial da equipe de suporte?" inicialmente caíam em `OUT_OF_DOMAIN` porque "suporte" e "equipe" não estavam cobertos; o termo salarial correto (`LGPD_PROTECTION`) só passou a prevalecer depois de incluir esses termos no vocabulário.

### Pipeline de geração (`src/rag.py`)

`generate_rag_response()` integra política, busca híbrida e geração estruturada:

1. avalia a política pela pergunta isolada; se `recusar`, retorna a recusa sem consultar `hybrid_search`;
2. busca os documentos relevantes e reavalia a política com os metadados das fontes recuperadas (para decidir `mascarar`);
3. monta o contexto (mascarado, se aplicável) e chama `llm.with_structured_output(RAGResponse)`;
4. valida que toda citação em `sources_used` corresponde a um chunk realmente recuperado (`filepath` + `chunk_id`) e que a `quotation` é um trecho literal do conteúdo desse chunk — evidências inventadas são rejeitadas;
5. em caso de erro de parsing, de validação Pydantic ou de evidência inválida, tenta novamente (`max_attempts`, padrão 3) enviando uma mensagem corretiva com o erro; se as tentativas se esgotam, propaga a causa original em `StructuredGenerationError`.

O `llm` é injetado (deve implementar `with_structured_output`), o que permite usar qualquer chat model compatível com LangChain e testar o pipeline com um LLM fake, sem chamadas reais de API.

### Testes de guardrails

`tests/test_guardrails.py` cobre, com pelo menos duas perguntas por caso:

- duas perguntas de dados pessoais (CPF, salário) recusadas por `LGPD_PROTECTION`, sem chegar a consultar o índice;
- duas perguntas de credenciais (senha, cartão) recusadas por `CREDENTIAL_PROTECTION`;
- duas perguntas fora do domínio recusadas por `OUT_OF_DOMAIN`;
- duas perguntas com fontes `sensitivity="restrito"`, confirmando que o dado sensível não aparece nem no prompt enviado ao LLM nem na resposta final;
- duas perguntas permitidas, validando o schema retornado e a correspondência literal da evidência com o chunk de origem.

### Executar a suíte completa

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Resultado obtido ao final da Etapa 3:

```text
81 passed, 16 subtests passed
```

Ao final da Etapa 4, após as correções diagnosticadas em [`RELATORIO.md`](RELATORIO.md) e os testes de regressão que as acompanham:

```text
130 passed, 22 subtests passed
```

## Etapa 4 — Avaliação (RAG Triad), interface e relatório

### Configurar o provedor de LLM

As etapas 1 e 2 rodam sem chave de API (embeddings locais). A partir da Etapa 3, a geração de respostas exige um LLM. Copie o arquivo de exemplo e preencha a chave:

```powershell
copy .env.example .env
```

O `.env` **nunca** é commitado (já está no `.gitignore`). As variáveis lidas por `config.py` são:

| Variável | Obrigatória | Descrição |
|---|---|---|
| `LLM_PROVIDER` | não | `openrouter` (padrão), `groq` ou `anthropic` |
| `OPENROUTER_API_KEY` | se `LLM_PROVIDER=openrouter` | Chave em [openrouter.ai/keys](https://openrouter.ai/keys) |
| `GROQ_API_KEY` | se `LLM_PROVIDER=groq` | Chave gratuita em [console.groq.com/keys](https://console.groq.com/keys) |
| `ANTHROPIC_API_KEY` | se `LLM_PROVIDER=anthropic` | Chave em [console.anthropic.com](https://console.anthropic.com) |
| `LLM_MODEL` | não | Padrão por provedor: `nex-agi/nex-n2.5-pro:free` (OpenRouter), `openai/gpt-oss-120b` (Groq) ou `claude-opus-5` (Anthropic) |
| `RETRIEVAL_K` | não | Chunks recuperados por pergunta (padrão `5`) |
| `MAX_GENERATION_ATTEMPTS` | não | Tentativas de geração estruturada (padrão `3`) |

O provedor padrão é o OpenRouter, com o modelo gratuito fixo `nex-agi/nex-n2.5-pro:free`, que não exige cartão de crédito. O roteador `openrouter/free` não é usado: ele escolhe um modelo diferente a cada chamada, e parte deles falha na saída estruturada. A troca de provedor é só uma variável de ambiente: `src/rag.py` recebe o modelo já construído e não conhece nenhum SDK específico.

Observações práticas sobre o OpenRouter:

- **O limite diário é da conta, não do modelo.** Contas que nunca compraram créditos têm cerca de 50 requisições gratuitas por dia, somando todos os modelos `:free`. Um ciclo completo de medição (benchmark + juiz) passa desse limite; para a interface de demonstração ele sobra.
- **Saldo negativo bloqueia até os modelos gratuitos.** Confira em [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits) antes de executar.
- **Nem todo modelo `:free` serve.** O pipeline usa `with_structured_output`, que exige suporte a *tool calling*. Verifique o campo `supported_parameters` em `https://openrouter.ai/api/v1/models`.

Para saber quais modelos têm cota disponível no momento, sem gastar cota relevante:

```powershell
.\venv\Scripts\python.exe -m eval.check_providers
```

As medições registradas em [`RELATORIO.md`](RELATORIO.md) foram feitas na Groq com `openai/gpt-oss-120b`. Para reproduzi-las, use `LLM_PROVIDER=groq`.

> **Nota para Windows:** se aparecer `SSL: CERTIFICATE_VERIFY_FAILED` ao chamar a API, o antivírus ou proxy está interceptando o certificado e o Python não reconhece a autoridade. Instale `pip install pip-system-certs` para que o Python use o repositório de certificados do próprio Windows.

### Executar o benchmark

Requer o índice FAISS criado (`python -m src.vectorstore`):

```powershell
.\venv\Scripts\python.exe -m eval.run_benchmark
```

Executa todas as perguntas de `benchmark/questions_and_ground_truth.json`, grava `eval/results.json` e imprime a tabela resumo por categoria. O executor não assume quantidade fixa de perguntas e salva o progresso a cada pergunta.

Cada execução também é arquivada com data em `eval/runs/`, para que uma reexecução não sobrescreva uma medição anterior — isso já aconteceu uma vez e custou uma medição boa. Erros de rate limit (HTTP 429) são repetidos com espera crescente, para que um estouro de cota não seja contado como defeito do pipeline.

> **Cota do tier gratuito.** A Groq limita 200.000 tokens por dia **por modelo**, recarregados continuamente (cerca de 8.300 por hora), e não zerados num horário fixo. Medido pelo tamanho real dos prompts, uma execução do benchmark consome cerca de 89.000 tokens e uma do juiz cerca de 33.000 — **cerca de 123.000 por ciclo completo, menos de dois ciclos por dia**. Ao esgotar a cota de um modelo, é possível trocar `LLM_MODEL` por outro (por exemplo `openai/gpt-oss-20b`), que tem cota própria; nesse caso os resultados deixam de ser diretamente comparáveis aos de execuções anteriores.

### Calcular a RAG Triad

```powershell
.\venv\Scripts\python.exe -m eval.judge_prompt
```

Lê o `eval/results.json` e grava `eval/triad_scores.json` com Context Relevance (determinística), Answer Relevance e Groundedness (LLM-as-judge). O juiz avalia 4 perguntas em paralelo e reaproveita vereditos de respostas que não mudaram — cada veredito guarda a impressão digital da resposta que julgou, então uma resposta nova é sempre reavaliada. Cada execução também é arquivada em `eval/runs/`.

### Calcular a nota pela rubrica do guia

```powershell
.\venv\Scripts\python.exe -m eval.score
```

Aplica a rubrica de 1,0 ponto por questão (0,5 resposta correta, 0,3 citação no arquivo certo, 0,2 coerência entre `confidence_level` e `is_refusal`) e imprime a nota por questão e por categoria. Questões que o juiz não conseguiu avaliar (por exemplo, por estouro de cota) são contadas à parte, e a nota é exibida como intervalo em vez de tratá-las como zero.

### Medir a recuperação sem LLM

```powershell
.\venv\Scripts\python.exe -m eval.retrieval_score
```

Calcula a Context Relevance comparando as fontes recuperadas com as `expected_sources` do gabarito, sem chamar LLM e sem gastar cota. Foi a ferramenta usada para validar as mudanças de recuperação descritas no relatório.

### Resultados

| Métrica | Valor |
|---|---:|
| Nota pela rubrica | **71,9%** (17,25 / 24) |
| Context Relevance | 0,89 |
| Answer Relevance | 0,54 |
| Groundedness | 0,73 |

Medido com `openai/gpt-oss-120b` na Groq, com as 24 perguntas avaliadas. O diagnóstico das falhas, a evolução da nota e as limitações da avaliação estão em [`RELATORIO.md`](RELATORIO.md).

### Interface de demonstração

A interface usa Streamlit e conversa com o pipeline real (busca híbrida, política de LGPD e geração estruturada validada por Pydantic).

O Streamlit já está em `requirements.txt`; se o ambiente foi criado antes desta etapa, reinstale as dependências:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Execução:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

A aplicação abre em `http://localhost:8501`. Antes de rodar, confirme que o índice FAISS existe (`python -m src.vectorstore`) e que o `.env` está preenchido.

Recursos da interface:

- histórico de conversa com `st.session_state` e `st.chat_message`;
- botões com perguntas de exemplo, incluindo uma que dispara o guardrail de LGPD;
- para cada resposta: nível de confiança, evidências citadas (arquivo + `chunk_id` + trecho literal) e o raciocínio;
- recusas aparecem destacadas, com o motivo traduzido;
- o índice FAISS e o modelo de embeddings são carregados uma única vez por sessão (`st.cache_resource`), para a demo não travar a cada pergunta.
- **Repetir cada medição três vezes**, para separar efeito real de variação do modelo.
- **Q02:** exige ajuste na representação vetorial dos arquivos estruturados.
