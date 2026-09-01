# Mini Desafio RAG — VendeFácil Knowledge Base

Projeto desenvolvido para o Mini Desafio de Assistente RAG Corporativo da VendeFácil Tecnologia Ltda.

## Contexto

A VendeFácil fornece soluções de automação comercial para pequenos e médios varejistas. O projeto constrói uma base de conhecimento capaz de processar fontes internas da empresa, recuperar informações relevantes e, nas próximas etapas, responder perguntas com evidências e proteção de dados.

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
├── src/
│   ├── ingest.py
│   ├── metadata.py
│   ├── sanity_check.py
│   ├── vectorstore.py
│   └── loaders/
│       ├── structured.py
│       ├── log_loader.py
│       ├── jsonl_loader.py
│       ├── markdown_loader.py
│       ├── pdf_loader.py
│       └── text_loader.py
├── tests/
│   ├── test_structured.py
│   ├── test_textual_loaders.py
│   ├── test_log_loader.py
│   ├── test_ingest.py
│   └── test_vectorstore.py
├── starter/
│   └── requirements.txt
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
.\venv\Scripts\python.exe -m pip install -r starter/requirements.txt
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

1. o Query Analyzer baseado em regras extrai `state`, `module`, `customer_id` e `priority`;
2. os filtros são normalizados e validados contra os valores realmente presentes no índice;
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

## Testes

Para executar todos os testes automatizados:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Resultado esperado após a Etapa 2:

```text
36 passed, 2 skipped
```

## Próximas etapas

- **Etapa 3:** síntese estruturada com Pydantic e guardrails de LGPD;
- **Etapa 4:** benchmark, relatório de falhas e interface de demonstração.
