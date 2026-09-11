# Relatório de Avaliação — Etapa 4

Assistente RAG Corporativo VendeFácil · Grupo 07 — Neves / Fregulha

## 1. Metodologia

| Item | Valor |
|---|---|
| Dataset | `benchmark/questions_and_ground_truth.json` — **24 perguntas** |
| Provedor / modelo | Groq · `openai/gpt-oss-120b` |
| Recuperação | Busca híbrida Dense (FAISS) + BM25Plus, fusão RRF (`k=60`), `RETRIEVAL_K=5` + 3 vagas sem filtro |
| Índice | 5.718 chunks |
| Execução | `python -m eval.run_benchmark` → `python -m eval.judge_prompt` → `python -m eval.score` |

O arquivo distribuído contém 24 perguntas, e não 20 como descrito no guia. O executor lê a quantidade do próprio arquivo, então uma versão com 20 roda sem alteração de código.

**Composição da medição final (10/09).** O benchmark rodou as 24 perguntas com o código atual; a Q22 falhou por estouro da cota diária e foi regenerada horas depois, com o mesmo código e o mesmo modelo, antes do juiz. Nenhum arquivo de código mudou entre as duas gerações. Todos os 24 vereditos vêm do mesmo juiz (`gpt-oss-120b`) — cinco foram reaproveitados de respostas byte a byte idênticas, verificadas pela impressão digital. Arquivos: `eval/runs/results-20260911T003303Z.json` e `eval/runs/triad-20260911T003338Z.json`.

### Como a nota é apurada

Rubrica do guia, 1,0 ponto por questão: **0,5** resposta correta (ou recusa correta), **0,3** citação apontando o arquivo certo, **0,2** `confidence_level` e `is_refusal` coerentes. O componente de correção usa o `answer_relevance` do LLM-as-judge (`Alta` = 0,5, `Média` = 0,25, `Baixa` = 0).

O cálculo está em `eval/score.py`, versionado. Ele distingue **questão não avaliada** de **questão zerada** — tratá-las como iguais já produziu uma nota falsamente baixa neste projeto (seção 4.5). Quando há questões sem avaliação, a nota é reportada como intervalo, não como número.

### Medição sem custo de API

`eval/retrieval_score.py` mede a qualidade da recuperação **sem chamar LLM nenhum**: compara as fontes recuperadas com as `expected_sources` do gabarito. É determinística, reproduzível e gratuita, e por isso foi a ferramenta usada para validar todas as mudanças de recuperação deste relatório. Duas hipóteses foram testadas e **descartadas** por ela sem gastar um token (seção 4.4).

## 2. Taxa de acerto

**17,25 / 24,00 pontos — 71,9%**, com as 24 questões avaliadas.

| Desfecho | Qtde |
|---|---:|
| Nota cheia (1,00) | 12 |
| Nota parcial (0,50–0,75) | 8 |
| Zero | 4 |

### Evolução ao longo do desafio

| | Estado inicial | Após correções de recuperação | Recuperação + medição (09/09) | **Final (10/09)** |
|---|---:|---:|---:|---:|
| **Pontuação** | 41,2% | 47,3% | 58,3% | **71,9%** |
| Context Relevance | 0,43 | 0,60 | 0,89 | **0,89** |
| Answer Relevance | 0,36 | 0,42 | 0,45 | **0,54** |
| Groundedness | 0,68 | 0,45 | 0,69 | **0,73** |
| Questões sem avaliação | — | 10 | 0 | **0** |

**Ressalva sobre as três primeiras colunas.** O resultado de 09/09 combina três mudanças: correção da recuperação, conserto da medição e retorno ao modelo `gpt-oss-120b` (execuções intermediárias usaram o `gpt-oss-20b`, por esgotamento de cota — ver seção 4.6). Não é possível separar a contribuição de cada uma. O ganho de recuperação, esse sim, está isolado e medido sem LLM: **0,654 → 0,886**.

**A última comparação é controlada.** Entre 09/09 e 10/09 mudou uma única coisa no código — a normalização de espaços na validação de citação (4.1) — com o mesmo modelo e o mesmo juiz. A comparação pergunta a pergunta atribui o ganho:

| Origem | Pontos |
|---|---:|
| Q03, Q09 e Q17 — de erro de validação para resposta aceita (correção 4.1) | **+2,75** |
| Q01 e Q19 — +0,25 cada, mesma resposta julgada Média → Alta | +0,50 |
| Demais 19 questões | 0,00 |
| **Total** | **+3,25** |

Nenhuma questão perdeu pontos. Os +0,50 fora das questões corrigidas estão dentro da variação esperada de um juiz LLM e não devem ser atribuídos a mudança alguma.

## 3. RAG Triad

| Métrica | Valor | Leitura |
|---|---:|---|
| **Context Relevance** | **0,89** | Métrica diagnóstica do guia. É também a única determinística, e a mais confiável do conjunto |
| **Answer Relevance** | **0,54** | Ainda o gargalo: o contexto certo chega, e parte das respostas sai incompleta |
| **Groundedness** | **0,73** | O sistema se apoia no que recuperou; não inventa |

### Desempenho por categoria

| Categoria | Qtde | Pontos | % |
|---|---:|---:|---:|
| Guardrails & LGPD | 6 | 5,75 | **95,8%** |
| Razão & Solução de Problemas | 4 | 3,00 | **75,0%** |
| Múltiplas Fontes (Multi-hop) | 3 | 2,25 | **75,0%** |
| Filtragem por Metadados | 4 | 2,75 | 68,8% |
| Fácil (RAG Básico) | 5 | 3,00 | 60,0% |
| Políticas Internas | 2 | 0,50 | 25,0% |

"Razão & Solução de Problemas" era a pior categoria (5,0%) e chegou a 75,0% — resultado direto da correção descrita em 4.3. "Fácil" (35% → 60%), "Multi-hop" (41,7% → 75%) e "Guardrails" (83,3% → 95,8%) subiram com a correção de citação de 4.1, que alcançou Q03, Q09 e Q17.

## 4. Diagnóstico das falhas

Na medição de 09/09 havia sete questões zeradas, distribuídas em quatro causas. Três delas (4.1) eram **defeito nosso, não do modelo**, e foram corrigidas e remedidas. Restam **quatro zeros**, com causa identificada em cada um: Q04 e Q12 (trecho certo fora do contexto, 4.2), Q21 (resposta ausente do acervo, 4.2) e Q02 (embedding, 4.7).

### 4.1 Citação literal rejeitada por diferença de espaçamento — 3 questões (corrigida)

**Afeta:** Q03, Q09, Q17. **Origem:** Etapa 3 (`src/rag.py`, `_validate_evidence`).

O erro registrado é idêntico nas três:

```
EvidenceValidationError: A quotation do chunk 'md-reembolso-secao-2-parte-1'
nao e um trecho literal.
```

O modelo cita o chunk **correto** e copia o trecho **certo**. O que reprova é a comparação: `_validate_evidence` exige que a citação seja subtrecho exato de `page_content`, e os chunks contêm quebras de linha e marcação markdown (`##`, `**`, crases). Ao reproduzir um trecho que atravessa uma quebra de linha, o modelo normaliza o espaçamento — e a comparação exata falha.

O propósito da regra é impedir evidência inventada, e esse propósito continua atendido se a comparação normalizar espaços em branco antes de comparar. Hoje ela rejeita respostas corretas por diferença de formatação.

**Correção aplicada:** normalizar espaços em branco dos dois lados antes da verificação de subtrecho. É localizada e não enfraquece a garantia: um trecho inventado continua reprovado.

**Medida em 10/09**, primeiro de forma dirigida (só as três questões afetadas: 3/3 passaram a responder) e depois no ciclo completo, com o mesmo modelo e o mesmo juiz de 09/09: **Q03 0 → 1,00, Q09 0 → 1,00, Q17 0 → 0,75**. São **+2,75 pontos**, a maior parte da subida de 58,3% para 71,9% (seção 2).

**Nota de método:** essa causa era invisível até hoje. O erro real ficava no `__cause__` da exceção e o benchmark registrava apenas *"não gerou resposta válida em 3 tentativas"*. Instrumentar cada tentativa foi o que tornou o diagnóstico possível — e revelou que as três "falhas de geração" eram, na verdade, a mesma falha nossa.

### 4.2 Recusa com o arquivo certo no contexto — 3 questões (hipótese refutada)

**Afeta:** Q04, Q12, Q21. **Origem real:** Etapa 1 (chunking) em Q04 e Q12 — o trecho com a resposta não chega ao contexto; Q21 não é falha do pipeline (resposta ausente do acervo).

Nas três, o arquivo esperado aparece entre as fontes recuperadas e o modelo recusa alegando `SEM_EVIDENCIA`. O guia é explícito sobre o custo: *"Recusar uma pergunta legítima vale zero, igual a errar."*

**Primeira hipótese — refutada.** Supusemos que o modelo recusava por comodidade, e implementamos a proibição por código: `_validate_evidence` rejeitava `SEM_EVIDENCIA` sempre que houvesse documentos no contexto. Validação em 10/09, no `gpt-oss-120b`: **0/3**. Q21 insistiu na recusa nas três tentativas; Q04 e Q12 tentaram contornar a regra e esbarraram em outras validações. A regra foi **revertida**.

**O que a inspeção do contexto mostrou** (sem LLM, listando os chunks entregues ao modelo):

| Pergunta | Arquivo esperado | Chunk que chegou ao modelo | Onde está a resposta |
|---|---|---|---|
| Q04 | `home_office.md` | só o título do documento | seção 1, "Modelos de Trabalho" |
| Q12 | `2026-02-engineering_outage_retrospective.md` | só o cabeçalho (data e participantes) | seção "Ações e Decisões Aprovadas" |
| Q21 | `beneficios_e_viagens.md` | seção 2, "Reembolso de Viagens e Despesas Comerciais" | **em lugar nenhum do acervo** |

**Q21 não tem resposta no acervo.** O gabarito espera "reembolso de até 80%" e "limite anual de R$ 2.500,00" para cursos. O arquivo indicado tem apenas duas seções (cartão benefícios e viagens) e nenhum outro documento, em nenhum formato, menciona reembolso de cursos ou certificações. Recusar com `SEM_EVIDENCIA` é o comportamento correto do pipeline — a nota zero nessa questão mede uma divergência entre gabarito e acervo, não uma falha nossa. Registramos sem alterar o benchmark.

**A recusa era honesta nas três.** Em Q04 e Q12 o contexto continha o arquivo certo, mas não o trecho com a resposta; em Q21 a resposta não existe. Além disso, a busca nunca devolve contexto vazio a esse ponto do pipeline — então a regra tornava **toda** recusa `SEM_EVIDENCIA` impossível e convertia recusa honesta em erro na interface.

**Por que não vimos antes:** a Context Relevance deste relatório é medida por **arquivo** (seção 6). Para essas três questões ela marcava acerto, e esse acerto sustentou a hipótese errada. É o custo concreto da limitação.

**Correção real para Q04 e Q12, não aplicada:** recuperação no nível de chunk — chunks de título isolados (sem conteúdo) competem com as seções e ocupam a vaga; juntar o título à primeira seção no chunking eliminaria esse caso.

### 4.3 Filtro inferido eliminando a fonte correta — corrigida

**Afetava:** Q04, Q12, Q19 e a categoria "Razão & Solução de Problemas" inteira.

O Query Analyzer infere filtros da pergunta e os aplica como conjunção rígida. O problema medido: em várias perguntas o filtro **acerta a forma e erra a intenção**. "Política de home office para a equipe de Engenharia" vira `doc_type=employee`, e a resposta está em `home_office.md`. O documento correto era eliminado antes da fusão.

A medição que localizou a causa comparou a posição do arquivo correto com e sem filtro (nível de arquivo — ver 4.2 para o que isso deixou de fora):

| Pergunta | Posição na busca **sem** filtro | Presente no resultado filtrado |
|---|---:|---|
| Q04 | 1ª | não |
| Q12 | 1ª | não |
| Q19 | 1ª | não |

O chunk certo era o primeiro colocado numa busca aberta e desaparecia do resultado filtrado. A recuperação não era ruim — era sabotada pelo filtro.

**Correção aplicada.** Quando os filtros são inferidos, o resultado recebe até 3 documentos **adicionais** vindos da busca sem filtro. São posições extras: nenhum resultado filtrado é removido. O comportamento é opcional (`complementar_sem_filtro`) e usado apenas pelo pipeline de geração — o comparativo da Etapa 2 continua com filtragem estrita, preservando a garantia daquela etapa.

Efeito medido, sem uso de LLM:

| Vagas extras | Context Relevance | Acham a fonte |
|---:|---:|---:|
| 0 | 0,654 | 15/19 |
| 1 | 0,737 | 16/19 |
| 2 | 0,781 | 16/19 |
| **3** | **0,886** | **18/19** |

### 4.4 Duas hipóteses testadas e refutadas

Ambas foram implementadas, medidas e descartadas **sem gastar cota**, graças ao medidor determinístico.

**Tolerar campo ausente no filtro.** A ideia: um filtro `module=estoque` não deveria eliminar `products.json`, que simplesmente não possui esse campo. Resultado medido: Context Relevance **caiu** de 0,654 para 0,610. Afrouxar o filtro deixa passar documentos demais e dilui o topo, expulsando as fontes certas de Q07, Q11 e Q13. Restringir a tolerância apenas ao campo `module` deu o mesmo resultado.

**Fundir o ranking sem filtro por pontuação, com peso menor.** Não mudou nada. O filtro devolve 10+ chunks do mesmo arquivo e ocupa todas as posições; o primeiro colocado da busca aberta contribui `1/181` contra `2/61` de um chunk filtrado, e nunca alcança. Qualquer peso alto o bastante para competir também dilui.

Foi o que levou ao desenho de 4.3: reservar posições **adicionais** em vez de disputar por pontuação.

### 4.5 A medição contava questão não avaliada como zero

Numa das execuções, o LLM-as-judge falhou por estouro de cota em **14 das 24 questões**. O cálculo da nota tratava `answer_relevance` vazio como zero, e reportou **39,6%** — quando o valor real estava entre 39,6% e 60,4%. Seis questões que **responderam** foram contadas como erro porque ninguém as avaliou.

Isso não é detalhe de apresentação: durante horas o diagnóstico apontou "falha de geração" onde havia apenas ausência de medição, e decisões de código foram tomadas em cima disso.

**Corrigido em `eval/score.py`:** questões não avaliadas são contabilizadas à parte e a nota vira intervalo, com aviso explícito. O `eval/judge_prompt.py` também passou a avisar quando não consegue avaliar alguma questão.

### 4.6 Cota como limite real do projeto

O tier gratuito da Groq limita **200.000 tokens por dia, por modelo**, recarregados continuamente (≈ 8.300 por hora), e não zerados num horário fixo. Medido em 10/09 pelo tamanho real dos prompts, um ciclo completo custa **≈ 123.000 tokens** (benchmark ≈ 89.000, juiz ≈ 33.000) — menos de dois ciclos por dia. A estimativa anterior, de 70.000, ficou defasada quando as vagas extras de 4.3 aumentaram o contexto enviado por pergunta.

Em 10/09 o limite interrompeu mais uma medição: validações dirigidas consumiram cota antes do ciclo, e o benchmark esgotou o saldo na Q22. A medição foi completada regenerando só essa questão e julgando as pendentes (≈ 30.000 tokens, contra ≈ 123.000 de um ciclo novo) — composição descrita na seção 1.

Esse limite moldou o trabalho de formas concretas: uma execução foi invalidada por rate limit no meio; uma medição boa foi perdida ao ser sobrescrita por outra contaminada; e parte das medições intermediárias foi feita no `gpt-oss-20b` porque a cota do `120b` havia se esgotado, o que impede comparação controlada entre elas.

Três mudanças no código vieram daí, e todas continuam valendo:

- `run_benchmark.py` e `judge_prompt.py` arquivam cada execução em `eval/runs/` com data — nenhuma medição é destruída por outra;
- ambos aplicam backoff em erro 429, separando estouro de cota de defeito do pipeline;
- o juiz passou a avaliar em paralelo e a reaproveitar vereditos já emitidos, com verificação de impressão digital da resposta para não misturar a nota de uma resposta com o texto de outra. O tempo caiu de mais de 20 minutos para cerca de 4.

### 4.7 Uma questão fora do alcance da recuperação

**Afeta:** Q02. **Origem:** Etapa 1 (representação vetorial dos arquivos estruturados).

**Q02** ("Quem é o Tech Lead e a PM do VendeFácil Estoque?") é a única falha de recuperação restante. `products.json` e `employees.csv` **não aparecem nem na busca sem filtro nenhum** — não é o filtro que os elimina, é o embedding que não os aproxima da pergunta. Nenhum ajuste na camada de filtragem alcança esse caso.

## 5. O que seria feito com mais 4 horas

Em ordem de retorno, com o alcance de cada item medido:

1. **Chunking: não isolar títulos de documento (≈ 1h).** Alcança Q04 e Q12 (4.2), onde o chunk de título ocupa a vaga da seção com a resposta. Q21 não entra: a resposta esperada não existe no acervo. Exige reindexar e remedir a Context Relevance.
2. **Atacar as respostas parciais (≈ 1h30).** Oito questões respondem com a fonte certa e recebem nota parcial — é o maior bloco isolado. `Answer Relevance` em 0,54 com `Context Relevance` em 0,89 mostra que o problema não é mais o que chega ao modelo, e sim o que ele faz com isso: prompt de geração e, possivelmente, tamanho de chunk.
3. **Repetir cada execução três vezes (≈ 1h + cota de dois dias).** É a maior limitação metodológica restante. Sem repetição não se separa efeito de ruído. A ≈ 123.000 tokens por ciclo, três repetições não cabem num único dia de cota.
4. **Q02 (≈ 1h).** Exige mexer na camada de embedding ou na estratégia de chunking dos arquivos estruturados — o único caso que a recuperação atual não alcança.

## 6. Limitações desta avaliação

- **Correção medida por LLM-as-judge.** `answer_relevance` e `groundedness` vêm de um juiz automático, o mesmo modelo do pipeline. A **Context Relevance é a única métrica determinística**, e por isso a mais confiável — foi ela que validou todas as decisões de recuperação deste relatório. Evidência de estabilidade do juiz: as mesmas respostas de 09/09, julgadas por um juiz diferente (OpenRouter `nex-n2.5-pro`), deram 57,3% contra 58,3% — 1 ponto de diferença.
- **Context Relevance no nível de arquivo.** O gabarito fornece `expected_sources` como caminhos de arquivo, não `chunk_id`. Construir um gabarito de chunks exigiria decidir nós mesmos quais chunks contêm a resposta, o que introduziria viés próprio na métrica. O custo dessa escolha apareceu em 4.2: a métrica marcou acerto em três questões cujo contexto tinha o arquivo certo e o trecho errado, e isso sustentou uma hipótese que precisou ser revertida.
- **Groundedness julgada de forma global, não afirmação por afirmação.** O guia define a métrica como *"cada afirmação da resposta está sustentada por algum trecho citado?"*. Nosso juiz emite um único veredito sobre a resposta inteira. Uma implementação fiel decomporia a resposta em afirmações e verificaria cada uma — mais precisa, e com custo de cota proporcional ao número de afirmações.
- **Execução única por versão, modelo não determinístico.** Sustentam conclusão os movimentos grandes e sobretudo os **contáveis**, verificáveis em `eval/results.json` e `eval/runs/`: Context Relevance de 0,654 para 0,886, questões que acham a fonte de 15/19 para 18/19, questões sem avaliação de 10 para 0.
- **Troca de modelo entre execuções.** Medições intermediárias usaram `gpt-oss-20b` por esgotamento de cota; a final usou `gpt-oss-120b`. Comparações de pontuação entre elas misturam efeito de correção e efeito de modelo.
- **O nível "mascarar" não é exercitado pelo benchmark.** Nenhuma das 24 perguntas tem `expected_metadata` de mascaramento, embora o guia preveja uma. O nível existe e está coberto por testes em `tests/test_guardrails.py`.
- **Perguntas que esperam recusa não pontuam Context Relevance.** A política recusa antes de consultar o índice, então não há contexto a avaliar — entram como não aplicáveis, e não como zero.
