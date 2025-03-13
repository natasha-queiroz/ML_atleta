let atletas = []
let model
let minVelocidade, maxVelocidade, minResistencia, maxResistencia, minPontuacao, maxPontuacao

function adicionarAtleta() {
    const nome = document.getElementById("nome").value.trim()
    const velocidade = parseFloat(document.getElementById("velocidadeCadastro").value)
    const resistencia = parseFloat(document.getElementById("resistenciaCadastro").value)

    if (!nome || isNaN(velocidade) || isNaN(resistencia)) {
        alert("Por favor, insira um nome e valores válidos.")
        return
    }

    const pontuacao = (velocidade + resistencia) / 2
    atletas.push({ nome, velocidade, resistencia, pontuacao })

    atualizarListaAtletas()

    document.getElementById("nome").value = ''
    document.getElementById("velocidadeCadastro").value = ''
    document.getElementById("resistenciaCadastro").value = ''
}

function atualizarListaAtletas() {
    const divLista = document.getElementById("listaAtletas")
    divLista.innerHTML =  atletas.map(a => `<p>${a.nome} - Velocidade: ${a.velocidade}, Resistencia: ${a.resistencia}</p>`).join('')
}

// CLASSIFICA E AGRUPA
function classificarAtletas() {
    if (atletas.length < 3) {
        alert("Adicione pelo menos 3 atletas para classificar.")
        return
    }

    const kMeansClustersValue = Math.max(2, Math.min(5, parseInt(document.getElementById("kMeansClusters").value)))
    const kMeansResultado = kMeansClustering(kMeansClustersValue)

    atualizarTabelaGrupos(kMeansResultado)

}  // O agrupamento K-means éum método para organizar dados em grupos ou clusters, com base na similaridade.
function kMeansClustering(k = 3) { 
    let grupos = Array(k).fill().map(() => [])
    let centroides = atletas.slice(0, k).map(a => ({ ...a }))

    for (let i = 0; i < 10; i++) {
        grupos = Array(k).fill().map(() => [])
        atletas.forEach(a => {
            let distancias = centroides.map(c => Math.hypot(a.velocidade - c.velocidade, a.resistencia - c.resistencia))
            let grupo = distancias.indexOf(Math.min(...distancias))
            grupos[grupo].push(a)
        })

        centroides = grupos.map(g => g.length ? ({
            velocidade: g.reduce((sum, a) => sum + a.velocidade, 0) / g.length,
            resistencia: g.reduce((sum, a) => sum + a.resistencia, 0) / g.length
        }) : { velocidade: 0, resistencia: 0 })
    }

    return { grupos, centroides }
}
// joga os atle
function atualizarTabelaGrupos(kMeansResultado) {
    const tabela = document.getElementById("tabelaGrupos").getElementsByTagName('tbody')[0]
    tabela.innerHTML = ''

    kMeansResultado.grupos.forEach((grupo, index) => {
        grupo.forEach(a => {
            const row = tabela.insertRow()
            row.insertCell(0).textContent = a.nome
            row.insertCell(1).textContent = a.velocidade
            row.insertCell(2).textContent = a.resistencia
            row.insertCell(3).textContent = a.pontuacao.toFixed(2)
            row.insertCell(4).textContent = `Grupo ${index + 1}`
        })
    })
}

function normalizar(valor, min, max) {
    return (valor - min) / (max - min)
}

function desnormalizar(valor, min, max) {
    return valor * (max - min) + min
}



async function treinarRegressaoLinear() {
    if (atletas.length < 3) {
        alert("Adicione pelo menos 3 atletas para treinar o modelo.")
        return
    }

    minVelocidade = Math.min(...atletas.map(a => a.velocidade))
    maxVelocidade = Math.max(...atletas.map(a => a.velocidade))
    minResistencia = Math.min(...atletas.map(a => a.resistencia))
    maxResistencia = Math.max(...atletas.map(a => a.resistencia))
    minPontuacao = Math.min(...atletas.map(a => a.pontuacao))
    maxPontuacao = Math.max(...atletas.map(a => a.pontuacao))

    modelLinear = tf.sequential()
    modelLinear.add(tf.layers.dense({ units: 1, inputShape: [2] }))
    modelLinear.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(0.1) })

    const xs = tf.tensor2d(atletas.map(a => [
        normalizar(a.velocidade, minVelocidade, maxVelocidade),
        normalizar(a.resistencia, minResistencia, maxResistencia)
    ]))

    const ys = tf.tensor2d(atletas.map(a => [normalizar(a.pontuacao, minPontuacao, maxPontuacao)]))

    await modelLinear.fit(xs, ys, { epochs: 200 })

    document.getElementById("resultado").innerText = "Modelo treinado! ✅"
}

async function treinarRedeNeural() {
    if (atletas.length < 3) {
        alert("Adicione pelo menos 3 atletas para treinar o modelo.")
        return
    }

    modelNN = tf.sequential()
    modelNN.add(tf.layers.dense({ units: 10, inputShape: [2], activation: 'relu' }))
    modelNN.add(tf.layers.dense({ units: 1 }))
    modelNN.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(0.01) })

    const xs = tf.tensor2d(atletas.map(a => [
        normalizar(a.velocidade, minVelocidade, maxVelocidade),
        normalizar(a.resistencia, minResistencia, maxResistencia)
    ]))

    const ys = tf.tensor2d(atletas.map(a => [normalizar(a.pontuacao, minPontuacao, maxPontuacao)]))

    await modelNN.fit(xs, ys, { epochs: 200 })

    document.getElementById("resultado_rede_neural").innerText = "Modelo treinado! ✅"
}



function preverAtletaAleatorio() {
    if (!modelLinear && !modelNN) {
        document.getElementById("previsao").innerText = "Treine ao menos um modelo antes de fazer previsões!"
        return
    }

    if (atletas.length < 3) {
        document.getElementById("previsao").innerText = "Adicione pelo menos 3 atletas para gerar previsões!"
        return
    }

    const velocidade = Math.random() * (maxVelocidade - minVelocidade) + minVelocidade
    const resistencia = Math.random() * (maxResistencia - minResistencia) + minResistencia
    const entrada = tf.tensor2d([[normalizar(velocidade, minVelocidade, maxVelocidade), normalizar(resistencia, minResistencia, maxResistencia)]])

    let resultados = `Novo Atleta: Velocidade: ${velocidade.toFixed(2)}, Resistência: ${resistencia.toFixed(2)}\n`

    // Previsão usando Regressão Linear, se treinado
    if (modelLinear) {
        modelLinear.predict(entrada).data().then(data => {
            const pontuacaoLinear = desnormalizar(data[0], minPontuacao, maxPontuacao);
            resultados += `📉 Regressão Linear - Pontuação prevista:  ${pontuacaoLinear.toFixed(2)}\n`
            atualizarPrevisao(resultados)
        })
    }

    // Previsão usando Rede Neural, se treinado
    if (modelNN) {
        modelNN.predict(entrada).data().then(data => {
            const pontuacaoNN = desnormalizar(data[0], minPontuacao, maxPontuacao)
            resultados += `🧠 Rede Neural - Pontuação prevista: ${pontuacaoNN.toFixed(2)}`
            atualizarPrevisao(resultados)
        })
    }
}

// Função auxiliar para atualizar a exibição sem sobrescrever valores
function atualizarPrevisao(texto) {
    document.getElementById("previsao").innerText = texto
}


