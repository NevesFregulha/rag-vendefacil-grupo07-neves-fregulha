# Acompanhamento - Mini Desafio RAG VendeFácil

**Integrante 1:** Fernanda Fregulha - [@fregulha](https://github.com/fregulha)
**Integrante 2:** Ester Da Silva Antonio Nóbrega Neves - [@EsterNevess](https://github.com/EsterNevess)

**Repositório:** `rag-vendefacil-grupo07-neves-fregulha`

---

## Como preencher

- Um bloco por encontro, em **ordem cronológica** - o encontro mais recente vai no **fim** do arquivo.
- O relato individual é escrito **pelo próprio integrante**, em primeira pessoa. Não escreva pelo colega.
- Escrever entre **17:30 e 17:40**. `commit` + `push` até as **18:00**, mesmo que o dia não tenha fechado.
- Mensagem de commit: `acompanhamento: AAAA-MM-DD`

**Um relato útil responde:** o que eu implementei, qual decisão técnica eu tomei e por quê, onde travei, e como (ou se) resolvi.

<details>
<summary>Exemplo de relato individual bom × ruim</summary>

❌ *"Trabalhei na parte de ingestão junto com meu colega. Avançamos bastante e conseguimos carregar os arquivos."*

✅ *"Implementei os loaders de CSV e JSONL em `src/ingest.py`. Decidi serializar cada linha do `customers.csv` como frase em linguagem natural em vez de manter o formato separado por vírgula, porque nos primeiros testes de similaridade os chunks CSV crus não recuperavam nada - o embedding não separa campo de valor. Travei ~40 min no `tickets.jsonl`: o `state` estava indo para o texto do chunk mas não para os metadados, então o filtro voltava vazio. Resolvi movendo a extração para antes da criação do `Document`. Usei o Claude para gerar o esqueleto do parser de JSONL; ajustei o schema de metadados na mão."*

</details>

---

## Encontro 1 - 2026-08-24

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - Fernanda Fregulha

Fiz, junto com a Ester, a configuração inicial do projeto no Windows. Criamos o ambiente virtual usando Python 3.14.4 e adaptamos os comandos do guia para funcionar no PowerShell, já que alguns estavam escritos para Linux. Depois instalamos as bibliotecas iniciais do RAG, como LangChain, FAISS, Pydantic, Pypdf, BM25 e python-dotenv, e rodamos o `pip check` para confirmar que estava tudo certo. Também exploramos as pastas da base e vimos que teremos arquivos CSV, JSON, JSONL, Markdown, PDF e TXT. Como essa aula ficou focada na preparação, deixamos a criação das pastas e do `src/ingest.py` para a próxima aula. Usei o GitHub Copilot para adaptar os comandos para o Windows e entender os próximos passos.

### Relato individual - Ester da Silva Antonio Nóbrega Neves

Fiz a preparação do projeto junto com a Fernanda. Na minha máquina, acompanhei a criação e a ativação do ambiente virtual no Windows, além da instalação das bibliotecas que vamos usar no desafio. Conferimos o `pip check` e o ambiente ficou funcionando normalmente. Também exploramos a pasta `data` para entender os tipos de arquivo disponíveis e percebemos que cada formato vai precisar de um tratamento diferente na etapa de ingestão. Não começamos o código nesta aula, então combinamos de criar as pastas e iniciar o `src/ingest.py` no próximo encontro. Usamos o GitHub Copilot para entender os comandos e adaptar as instruções para o PowerShell.

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Configuramos o ambiente virtual Python e instalamos as dependências iniciais do projeto.
- Validamos o ambiente com `python` e `pip check`.
- Exploramos a organização dos dados e identificamos os formatos que serão processados.

**Ficou pendente:**
- Criar as pastas e arquivos iniciais da aplicação.
- Implementar os leitores e o chunking adaptativo para CSV, JSON, JSONL, Markdown, PDF e TXT.
- Atualizar o arquivo de dependências para registrar as bibliotecas instaladas.

**Bloqueios em aberto:**
- Não tivemos bloqueios durante a configuração do ambiente.

**Próximo passo (início do encontro 2):**
- Criar as pastas iniciais do projeto, atualizar o arquivo de dependências e começar o `src/ingest.py`, iniciando pelos leitores dos arquivos da base.

**Uso de assistentes de IA:**
- Utilizamos o GitHub Copilot para tirar dúvidas, entender os comandos e adaptar as instruções para o PowerShell do Windows. Também conferimos os resultados e validamos a instalação com `pip check`.

---

## Encontro 2 - 2026-08-26

**Etapa:** 1 - Ingestão heterogênea, metadados e indexação vetorial

### Relato individual - Fernanda Fregulha

Preparei a estrutura compartilhada do projeto em `src/`, `src/loaders/` e `tests/` e implementei em `src/metadata.py` o contrato comum dos metadados, com validação de `source_file`, `doc_type`, `chunk_id` e `sensitivity`, incluindo detecção de IDs duplicados. Depois implementei os loaders de JSONL, Markdown, PDF e TXT com chunking adaptativo: um ticket por chunk no JSONL, divisão por seção nos Markdown, divisão por página e parágrafo nos PDFs e separação de mensagens nos e-mails. Validei 203 chunks textuais e classifiquei como `restrito` cinco e-mails internos que continham credenciais. Criei testes automatizados para os metadados e loaders; os oito testes executados passaram. Também revisei a ingestão estruturada depois do merge e identifiquei que `products.json` e `stores.json` foram tratados como um objeto único e que `system_logs.csv` ainda não foi incluído, deixando 503 registros pendentes. Usei o Codex para gerar esqueletos, revisar os formatos reais da base e executar os testes; conferi os resultados, ajustei as estratégias de chunking e preservei loaders diferentes para cada natureza de documento.

### Relato individual - Ester da Silva Antonio Nóbrega Neves

> A Ester deve escrever aqui o próprio relato individual antes do encerramento da entrega.

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
- Estrutura inicial de `src/`, `src/loaders/` e `tests/`, com contrato e validação de metadados.
- Loaders de JSONL, Markdown, PDF e TXT com chunking adaptativo, totalizando 203 chunks textuais validados.
- Loader inicial de CSV e JSON, com 5.012 chunks estruturados atualmente carregados.
- Testes automatizados de metadados, loaders textuais e ingestão estruturada.
- Pull Requests das TASKS 01, 02 e 03 integrados à `main`, preservando os commits individuais.

**Ficou pendente:**
- Corrigir a leitura das listas internas de `products.json` e `stores.json`, que devem gerar 5 e 50 chunks, respectivamente.
- Incluir os 450 registros de `data/semi_structured/system_logs.csv` com `doc_type="log"`.
- Integrar todos os loaders em `src/ingest.py`, criar e persistir o índice FAISS e executar o script de sanidade.
- Atualizar o `README.md` com as instruções e decisões da Etapa 1.

**Bloqueios em aberto:**
- A ingestão estruturada está 503 chunks abaixo do total esperado. Decidimos registrar a pendência e corrigi-la antes de gerar o índice FAISS.

**Próximo passo:**
- Corrigir a ingestão de produtos, lojas e logs; em seguida, executar a TASK 04 para integrar os loaders, gerar o FAISS e validar as três perguntas de sanidade.

**Uso de assistentes de IA:**
- Utilizamos o Codex para orientar o fluxo de branches e Pull Requests, propor os esqueletos dos loaders, revisar os formatos reais dos arquivos e automatizar testes. As sugestões foram conferidas contra o corpus, e ajustamos manualmente as estratégias de chunking, os metadados e a classificação de sensibilidade. A revisão também permitiu identificar os 503 registros ainda ausentes da ingestão estruturada.

---

## Encontro 3 - AAAA-MM-DD

**Etapa:** 3 - Síntese estruturada, evidência e guardrails de LGPD

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Próximo passo (início do encontro 4):**
-

**Uso de assistentes de IA:**
-

---

## Encontro 4 - AAAA-MM-DD

**Etapa:** 4 - Avaliação (RAG Triad), interface e relatório

### Relato individual - [Nome do Integrante 1]

### Relato individual - [Nome do Integrante 2]

### Resumo do dia (escrito em conjunto)

**Entregamos hoje:**
-

**Ficou pendente:**
-

**Bloqueios em aberto:**
-

**Preparação para o Demo Day:**
-

**Uso de assistentes de IA:**
-

---

*TIC em Trilhas · PUC-Rio · Instituto ECOA · MCTI Futuro · Softex*
