# Manutenção do Streamlit — 12–13/11/2026

Build atual: `streamlit-20261113.2` (compatibilidade da busca e incertezas).
Build anterior: `streamlit-20261113.1` (correção das regressões da interface).
Build inicial da manutenção: `streamlit-20261112.1`.
Escopo: aplicação pública `gusmock/mono_polyauxic_kinetics`, derivada do commit
`2bc20a9f5e3e1b75b3ed862cbbccdba414772950`. O código da suíte institucional e os
arquivos históricos de reprodução não integram esta refatoração.

## Solicitação da informática

Chamado **TN-2026111186000044**, prazo **13/11/2026**. O relatório aponta
DOMPurify 3.2.6 e Moment 2.30.1. Mínimos solicitados: 3.4.17 e 2.31.0,
respectivamente. A fila pública de issues estava sem chamados abertos na consulta.

O `requirements.txt` fixa Streamlit **1.67.0** e suas dependências Python diretas
e transitivas. Removidos `streamlit[auth]` e Authlib. Python **≥3.12** é necessário
para NumPy 2.5.2 e SciPy 1.18.0; o ambiente de validação usa Python 3.12.
Não instalar estes requisitos no ambiente de outra aplicação.

Foram inspecionados o lockfile oficial e os arquivos JavaScript do wheel instalado:

| Biblioteca | Versão verificada | Arquivo distribuído |
| --- | --- | --- |
| DOMPurify | 3.4.17 | `purify.es.BXhPiKO5.js` |
| Moment | 2.31.0 | `moment.4ragD6OA.js` |

Wheel Streamlit: `streamlit-1.67.0-py3-none-any.whl`.
SHA-256: `f29da87c443bfbdfc52738616e3d65745d438e718da8faf69e2be0da1e52ce64`.
O verificador `scripts/check_legacy_frontend.py` falha diante de versões antigas,
ausentes ou implementações reconhecidas cuja versão não possa ser determinada.
A verificação é específica dessas bibliotecas; não substitui um scanner geral.

## Correções da refatoração inicial (12/11/2026)

As decisões de busca numérica, exibição das incertezas e orçamento total abaixo
foram revistas no build `streamlit-20261113.2`, conforme registro ao final.

- Removidos login, cadastro, coleta de perfil e gravação automática de uploads.
  Arquivos antigos existentes no servidor não são apagados.
- Separados entrada, núcleo matemático, orquestração, tradução, figuras,
  exportação e interface. O legado funciona sem importar a suíte privada.
- Resultados conservados apenas na sessão: idioma, fonte, eixos e downloads
  não repetem o cálculo. Dados, configurações científicas ou interpretação do
  arquivo alterados invalidam o resultado anterior.
- Leitura limitada de CSV/XLSX, separadores de colunas e decimal automáticos
  com opção de escolha manual, rejeição de erros
  em pares, infinitos, fórmulas, macros, links externos e descompactação excessiva.
  Nenhuma linha parcialmente inválida é descartada silenciosamente.
- Ordenação estável mantém a identidade das observações com tempos repetidos
  entre ajuste, máscara de exclusão, gráfico e exportação. Tempos distintos
  não são mais agrupados após arredondamento a quatro casas.
- Corrigida a amplitude da primeira ordem: `k = r_max/(p*abs(y_f-y_i))`.
  A derivada inicial passa a corresponder a `r_max` na resposta crescente.
- Removida a redundância translacional do softmax: último logit fixo em zero.
  Assíntota inicial fixa e lag não utilizado da primeira ordem são excluídos da
  otimização. Contagem explícita de parâmetros livres e da variância residual.
- Covariância local calculada a partir do Jacobiano dos resíduos. A antiga
  pseudoinversa da Hessiana penalizada era singular pelo softmax e omitia o fator
  dois da Hessiana da soma de quadrados. Limites ativos, falta de posto ou
  convergência incompleta impedem apresentar erros-padrão como confiáveis.
- Mantida seleção pelo primeiro mínimo local. Candidatos com conjuntos distintos
  de observações retidas não são ordenados automaticamente, mesmo com o mesmo N.
  Falhas de convergência e critérios não finitos são excluídos da seleção.
- A primeira ordem monofásica pode ser reutilizada nas duas famílias sem novo
  ajuste, mas continua sendo considerada na comparação de ambas.
- A restrição de assíntota final zero foi retirada: ela não implementava uma
  cinética decrescente válida com essas equações e taxas positivas.
- Removidos scripts externos de badges. Recuperados o cartão autoral, foto,
  afiliação, sete perfis profissionais e referências completas por links.
- Exportação com dados brutos, candidatos, parâmetros numéricos, SE, resíduos,
  exclusões, opções, semente, horário UTC, SHA-256 da entrada e build. Strings
  não são convertidas em fórmulas ou links executáveis no Excel.
- Figuras são fechadas e seu uso é serializado para evitar interferência entre
  sessões. Limites por arquivo, por análise e de concorrência reduzem consumo
  acidental de recursos após a abertura do acesso.

As correções podem alterar erros-padrão, critérios e modelos selecionados.
Resultados históricos devem ser preservados com sua versão; não são reescritos.

## Instalação institucional

**O `git pull` deve ser feito na branch `main` do repositório público.** Ele
atualiza o código, mas não instala pacotes nem troca o Python do serviço.
Se o ambiente ainda utilizar Python 3.9, criar um novo ambiente Python 3.12,
instalar os requisitos e apontar o `ExecStart` do serviço para o novo executável.
Não basta reinstalar os requisitos dentro de um ambiente Python 3.9.

Exemplo de preparação, preservando o ambiente antigo para retorno:

```bash
git pull --ff-only origin main
python3.12 -m venv venv-py312
venv-py312/bin/python -m pip install --upgrade pip
venv-py312/bin/python -m pip install --upgrade -r requirements.txt
venv-py312/bin/python -m pip check
venv-py312/bin/python scripts/check_legacy_frontend.py
```

Ajustar o serviço para usar o caminho absoluto de `venv-py312/bin/python -m
streamlit run app.py`, mantendo diretório de trabalho, porta e proxy existentes.
Após editar a unidade: `systemctl daemon-reload` e
`systemctl restart streamlit-mono_polyauxic_kinetics`.
A informática deve executar essas etapas com as permissões administrativas
habituais; o programa não altera a unidade nem reinicia serviços automaticamente.

1. Confirmar Python 3.12, diretório do checkout e executável utilizado pelo
   serviço `streamlit-mono_polyauxic_kinetics`. O Python do shell pode ser diferente.
2. Preservar o checkout anterior e registrar `python -m pip freeze` do ambiente
   atualmente utilizado. Criar um ambiente separado se o Python antigo for <3.12.
3. Atualizar o checkout com a revisão revisada desta manutenção. Instalar com o
   **mesmo Python do serviço**, não com um `pip` global:

   ```bash
   /CAMINHO/DO/AMBIENTE/bin/python -m pip install --upgrade -r requirements.txt
   /CAMINHO/DO/AMBIENTE/bin/python -m pip check
   /CAMINHO/DO/AMBIENTE/bin/python scripts/check_legacy_frontend.py
   ```

4. Reiniciar o serviço após a instalação e conferir os logs e a página HTTPS.
   O acesso deve abrir diretamente. Validar o exemplo sintético, um upload
   autorizado e os downloads. Confirmar WebSocket e limites no proxy reverso.
5. Solicitar nova varredura externa do CSIRT no domínio. Se o navegador ou proxy
   mantiver assets antigos, conferir cache e a revisão realmente executada.
   **Uma atualização de requirements sem reiniciar não troca assets já servidos.**
6. Em caso de falha, restaurar checkout e ambiente anteriores como conjunto.
   Não apagar dados antigos nem reaproveitar segredos no repositório.

Este trabalho não comprova instalação no servidor ou liberação do chamado.
A notificação ao responsável e o encerramento dependem da verificação externa.

## Melhorias propostas

1. Adotar explicitamente um modelo para respostas decrescentes e tratar sinal,
   assíntotas e interpretação de taxas como parte do protocolo científico.
2. Validar o procedimento de outliers contra a referência metodológica desejada;
   o legado combina Charbonnier, MAD e BH e foi rotulado como aproximação.
3. Acrescentar perfil de verossimilhança/bootstrap para identificabilidade prática,
   intervalos de confiança e incerteza na seleção de fases.
4. Avaliar, como decisão metodológica separada, mínimo global versus primeiro
   mínimo local e a política de alternância AIC/AICc/BIC.
5. Se a demanda pública crescer, separar execução numérica em fila de processos
   com cancelamento e quotas por origem no proxy; atualmente o limite vale por
   processo e o orçamento é cooperativo dentro do otimizador.

## Fontes técnicas

- [Streamlit 1.67.0](https://github.com/streamlit/streamlit/releases/tag/1.67.0)
- [Lockfile JavaScript 1.67.0](https://github.com/streamlit/streamlit/blob/1.67.0/frontend/yarn.lock)
- [DOMPurify 3.4.17](https://github.com/cure53/DOMPurify/releases/tag/3.4.17)
- [Moment 2.31.0](https://github.com/moment/moment/releases/tag/2.31.0)
- [Equações de referência](https://doi.org/10.1007/s11538-026-01621-7)

## Verificação inicial (12/11/2026)

- **45 testes passaram** em Python 3.12: CSV/XLSX e limites, erros de entrada,
  recuperação de parâmetros em uma e duas fases, primeira ordem, mudança de
  unidades de tempo, remoção de outliers, critérios, concorrência, timeout,
  exportação e fluxo real de widgets com `streamlit.testing.v1.AppTest`.
- Upload → ajuste → gráficos → exportação conferidos em inglês, português e
  francês. Mudanças de apresentação não repetem o ajuste; novas condições e
  arquivos inválidos removem os resultados anteriores. Sessões são isoladas.
- Análise estática Ruff e verificação de whitespace concluídas sem erros.
- Verificador do JavaScript instalado: DOMPurify 3.4.17 e Moment 2.31.0 aprovados.
- Consulta OSV em 12/11/2026: nenhum alerta retornado para as 46 dependências do
  ambiente de validação e para Watchdog 7.1.0, específico de Linux/Windows.
  Isso representa a cobertura da base consultada, não prova ausência de falhas.
- Testes locais em macOS; a automação GitHub acrescenta execução nativa em Linux.
  A execução institucional e a varredura externa permanecem pendentes.

Os testes sintéticos exercitam os casos descritos. Não constituem validação
universal da identificabilidade, do número de fases ou do método de outliers.

## Correções solicitadas pelo autor (13/11/2026)

Relato recebido diretamente: perda das informações autorais, detecção automática
incompleta, parâmetros/erros padrão pouco visíveis, demora e diferenças de
comportamento. Nenhuma issue aberta encontrada no repositório público na consulta.
As correções abaixo pertencem exclusivamente ao legado público; não alteram
versão, build ou encerramento da suíte institucional rc.3.

| Item | Regressão constatada | Correção |
| --- | --- | --- |
| Autoria | Rodapé reduzido ao nome e siglas | Nome/título, foto, afiliação completa, localização, sete perfis e quatro referências científicas restaurados |
| Importação | Separador decimal manual, embora o delimitador já tivesse modo automático | Delimitador e decimal automáticos por padrão, inclusive valores textuais do XLSX; opções manuais preservadas |
| Resultados | Tabelas dentro de seção recolhida; quatro casas decimais ocultavam incertezas pequenas | Estimativas e erros padrão diretamente visíveis, seis algarismos significativos, identificação de valores fixos e não estimáveis |
| Outliers | Padrão alterado de robusto/FDR para nenhuma remoção | Padrão original robusto/FDR, Q = 1%, restaurado; as três opções continuam disponíveis |
| Acompanhamento | Só havia resultados após todos os ajustes | Progresso por família/fase, prévia dos candidatos concluídos e tempo da análise registrados |
| Seleção | Destaques e nomes das abas simplificados | Equações nos nomes das abas e destaques dos candidatos selecionados recuperados |

A importação é testada com vírgula, ponto e vírgula e tabulação como delimitadores,
ponto e vírgula decimais, cabeçalhos entre aspas, notação científica, linhas
vazias e réplicas de comprimentos diferentes. Convenções decimais misturadas
exigem correção ou escolha explícita; não se infere separador de milhares.
A convenção efetivamente usada também integra a auditoria do Excel.

Os testes da interface comparam os valores exibidos com o resultado numérico de
cada candidato, incluindo troca de candidato sem recalcular, três idiomas,
erros padrão pequenos, créditos e referências. Os parâmetros não ficam dentro
de expansores. Erro padrão ausente continua sinalizado como não estimável;
não foi substituído por zero nem recalculado por uma fórmula antiga inválida.

### Comparação de desempenho

Comparação no mesmo computador e ambiente Python 3.12, semente 42, robusto/FDR
com Q = 1%, entre o núcleo do commit original
`2bc20a9f5e3e1b75b3ed862cbbccdba414772950` e o publicado na manutenção
`6d84db845242aa3543cdb36cfdbc478a05c14c7b`. A revisão de 13/11 não altera o
algoritmo de otimização desse segundo núcleo. Medidas exploratórias únicas,
sem execução simultânea dos dois programas; não são limites de desempenho.

Dados sintéticos: 65 tempos igualmente espaçados de 0 a 30, duas fases,
`y_i=1`, `y_f=20`, `p=(0.4, 0.6)`, `r_max=(1.5, 1.2)`, `lambda=(2, 14)`;
ruído normal com desvio 0.08 e semente 10. No caso de 195 observações,
cada tempo aparece três vezes, com ruídos independentes.

Os tempos abaixo somam os candidatos sigmoides de 1 a 5 fases, cada um com
pré-ajuste robusto e ajuste final, na família indicada. Não incluem renderização,
transferência de arquivos, ajustes híbridos adicionais ou carga do servidor.

| Caso | Original | Manutenção publicada |
| --- | ---: | ---: |
| Gompertz, 65 observações | 17.361 s | 12.773 s |
| Boltzmann, 65 observações | 21.783 s | 13.752 s |
| Gompertz, 195 observações | 12.801 s | 11.431 s |

Não se reproduziu piora geral nesses casos. Houve candidatos mais lentos:
por exemplo, Gompertz de 3 fases/65 observações passou de 1.975 s a 3.053 s.
A busca estocástica e a parametrização independente podem mudar a trajetória
em modelos com fases excedentes. No ajuste de duas fases dos dados gerados
com duas fases, a SSE concordou entre as versões em menos de 4 × 10⁻⁸ nos
três casos. Isso não comprova equivalência de todos os resultados nem o tempo
na instalação institucional. Para reproduzir uma lentidão específica, são
necessários o arquivo utilizado, as opções e a revisão em execução.

### Diferenças científicas preservadas e explicitadas

A manutenção não é numericamente idêntica ao código anterior. Permanecem as
correções descritas acima: amplitude da primeira ordem, eliminação do parâmetro
redundante, contagem dos parâmetros independentes/variância, covariância pelo
Jacobiano e bloqueio de comparação entre candidatos que removem observações
diferentes. Elas podem alterar erros padrão, critérios e seleção de fases.
O botão de assíntota final zero continua ausente porque as equações anteriores
não implementavam corretamente uma curva decrescente. A revisão `streamlit-20261113.1`
não introduz novas mudanças nesses métodos nem reduz o orçamento de otimização
para produzir uma aparência de maior velocidade.

Na comparação adicional da sequência completa de ajustes numéricos (ambas as
famílias, máximo de 5 fases, mesmos 65 dados Gompertz), foram produzidos os mesmos
11 candidatos estruturais, incluindo a alternativa de primeira ordem. A execução
original levou **50.632 s** e a manutenção **37.440 s**. São tempos de uma execução
por versão nesse computador, sem renderização ou rede, e não uma medição da
instalação institucional. O acompanhamento da execução foi corrigido sem alterar
o núcleo matemático nesta revisão.

Verificação da correção: **59 testes passaram** em Python 3.12, incluindo fluxo
real de widgets nos três idiomas, parâmetros por candidato, leitura automática,
créditos, progresso, isolamento de sessões e exportação. Ruff aprovado. O
verificador dos arquivos JavaScript instalados confirmou novamente DOMPurify
3.4.17 e Moment 2.31.0. A automação do repositório repete esses testes em Linux.

## Compatibilidade da busca e incertezas — build streamlit-20261113.2

O autor confirmou em 13/11/2026 a prioridade de **compatibilidade com a estratégia
de busca anterior**, mantendo a atualização de dependências, acesso livre e
os diagnósticos de incerteza. A reprodução com dados autorizados revelou duas
regressões: a mudança das coordenadas da evolução diferencial podia levar a
outro mínimo, e um único parâmetro no limite eliminava todos os erros padrão.
Os dados experimentais, seus resultados e os arquivos de reprodução ficam fora
de ambos os repositórios. Os testes publicados usam exclusivamente dados sintéticos.

### Busca e seleção

- Restaurados a dimensão e as coordenadas originais da busca, os limites dos
  logits `[-10, 10]`, a população inicial de 50 indivíduos, a distribuição e a
  semente, `best1bin`, até 3.000 gerações, o polimento da evolução diferencial e
  o refinamento L-BFGS-B. Restaurada a tolerância absoluta `1e-6` do teste da
  primeira fase para considerar a alternativa de primeira ordem.
- A referência do softmax é normalizada **depois** do ajuste. O parâmetro
  redundante, os valores fixos e o lag não usado na primeira ordem continuam
  fora da contagem estatística e da covariância. A busca original e a estimativa
  da incerteza têm finalidades distintas.
- Se o segundo polimento falhar em uma solução já convergente da evolução
  diferencial, a solução convergente é conservada. Se ambos falharem,
  tenta-se um refinamento SLSQP com a ordem dos lags como restrição explícita.
  Isso trata a descontinuidade da penalidade antiga perto de lags coincidentes.
  O resultado só é aceito se convergir, respeitar a ordem e não piorar o objetivo
  além da tolerância numérica. A exportação registra o refinamento utilizado.
- Não se restaurou a aceitação silenciosa de resultados sem convergência ou de
  comparações entre observações diferentes. Permanecem os avisos e os bloqueios
  quando não há estimativa válida.
- Removido o corte padrão de 180 segundos da **análise inteira**, que prejudicava
  a conclusão de todas as fases em processadores mais lentos. Cada otimização
  mantém seu limite cooperativo de 120 segundos. Um chamador da API pode ainda
  fornecer explicitamente `time_budget` para limitar a análise completa.
  O progresso e os resultados já calculados continuam sendo apresentados.

Essa restauração da estratégia não promete identidade bit a bit: a correção da
amplitude da primeira ordem, os diagnósticos, a covariância, a contagem estatística
correta e o refinamento de recuperação permanecem. Não se reduziu a precisão nem
se descartaram réplicas para diminuir artificialmente o tempo.

### Erros padrão

Com convergência e Jacobiano de posto completo, a covariância local é calculada
por SVD nos parâmetros independentes. Quando há uma restrição ativa ou uma fronteira a menos de um passo das
diferenças finitas do Jacobiano, os erros
padrão calculáveis são apresentados e identificados como aproximações lineares
locais que **desconsideram a restrição**. Não representam intervalos de confiança
validados sob restrição. Rank deficiente ou ausência de convergência continuam
produzindo a indicação de erro não estimável. Valores fixos e o lag ausente da
primeira ordem são diferenciados de erros padrão iguais a zero.

O Excel inclui `uncertainty_status`, condicionamento do Jacobiano, restrições
ativas, estratégia de busca e refinamento. Assim, o aviso não se perde no download.

Referências metodológicas:

- [SciPy: covariance and linear approximation in curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- [SciPy: SLSQP and constrained convergence](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

### Verificação

**66 testes passaram** em Python 3.12 na validação local, incluindo o fluxo de
interface nos três idiomas. Acrescentados: regressão de busca contra um ajuste
sintético do código original; erro padrão em limite comparado a um Jacobiano
analítico independente; recuperação de falha do refinamento; manutenção de uma
análise completa que ultrapassa 180 segundos sem orçamento total explícito;
aviso de aproximação na interface; e diagnósticos correspondentes no Excel.
Resultados de uma versão anterior são invalidados automaticamente ao mudar
o build, evitando exibir cálculos antigos com os novos diagnósticos. Ruff
aprovado. A automação do GitHub repete os testes em Linux.
