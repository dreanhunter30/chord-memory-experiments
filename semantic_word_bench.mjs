import fs from "node:fs";
import path from "node:path";
import { env, pipeline } from "@huggingface/transformers";

const cwd = process.cwd();
env.cacheDir = path.join(cwd, ".cache", "huggingface");
env.useFSCache = true;

const concepts = [
  ["собака", "пёс", "домашнее животное, которое лает"],
  ["кошка", "кот", "домашнее животное, которое мяукает"],
  ["автомобиль", "машина", "транспорт с мотором и четырьмя колёсами"],
  ["врач", "доктор", "человек, который лечит больных"],
  ["ребёнок", "малыш", "маленький человек"],
  ["дом", "жилище", "место, где живут люди"],
  ["работа", "труд", "занятие, за которое получают зарплату"],
  ["грусть", "печаль", "тяжёлое чувство без радости"],
  ["радость", "счастье", "приятное чувство от хорошего события"],
  ["еда", "пища", "то, что люди едят"],
  ["помощь", "поддержка", "действие, облегчающее чужую задачу"],
  ["ошибка", "промах", "неправильный результат действия"],
  ["дорога", "путь", "полоса для движения между местами"],
  ["огонь", "пламя", "горячее свечение при горении"],
  ["холод", "мороз", "состояние с очень низкой температурой"],
  ["деньги", "средства", "то, чем оплачивают покупки"],
  ["компьютер", "ПК", "электронное устройство для обработки данных"],
  ["телефон", "смартфон", "карманное устройство для звонков"],
  ["учитель", "педагог", "человек, который обучает учеников"],
  ["книга", "издание", "текст на переплетённых страницах"],
  ["болезнь", "недуг", "нарушение здоровья организма"],
  ["лекарство", "препарат", "вещество для лечения болезни"],
  ["море", "водоём", "огромное пространство солёной воды"],
  ["лес", "чаща", "большая территория, покрытая деревьями"],
  ["самолёт", "аэроплан", "транспорт, который летает в небе"],
  ["поезд", "состав", "вагоны, движущиеся по рельсам"],
  ["музыка", "мелодия", "искусство организованных звуков"],
  ["сон", "сновидение", "отдых организма с закрытыми глазами"],
  ["страх", "боязнь", "чувство перед возможной опасностью"],
  ["любовь", "привязанность", "глубокое тёплое чувство к кому-либо"],
  ["время", "период", "то, что измеряют часами"],
  ["вода", "жидкость", "прозрачное питьё без цвета"],
  ["солнце", "светило", "звезда, освещающая Землю днём"],
  ["луна", "спутник", "небесное тело, обращающееся вокруг Земли"],
  ["одежда", "наряд", "вещи, которые надевают на тело"],
  ["обувь", "ботинки", "то, что надевают на ноги"],
  ["магазин", "лавка", "место, где продают товары"],
  ["больница", "клиника", "место, где лечат пациентов"],
  ["быстрый", "скорый", "движущийся с высокой скоростью"],
  ["смелый", "храбрый", "не боящийся опасности"],
];

const records = concepts.map(([target]) => target);
const synonymQueries = concepts.map(([target, synonym]) => [synonym, target]);
const definitionQueries = concepts.map(([target, , definition]) => [definition, target]);

function tensorRows(tensor) {
  const [count, dims] = tensor.dims;
  return Array.from({ length: count }, (_, index) =>
    Float32Array.from(tensor.data.slice(index * dims, (index + 1) * dims)),
  );
}

function dot(left, right) {
  let sum = 0;
  for (let index = 0; index < left.length; index++) {
    sum += left[index] * right[index];
  }
  return sum;
}

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function makeProjection(outputDims, inputDims, seed) {
  const random = rng(seed);
  const matrix = new Int8Array(outputDims * inputDims);
  for (let index = 0; index < matrix.length; index++) {
    matrix[index] = random() < 0.5 ? -1 : 1;
  }
  return matrix;
}

function phaseCode(vector, projectionX, projectionY, outputDims) {
  const code = new Uint8Array(outputDims);
  for (let output = 0; output < outputDims; output++) {
    let x = 0;
    let y = 0;
    const offset = output * vector.length;
    for (let input = 0; input < vector.length; input++) {
      x += vector[input] * projectionX[offset + input];
      y += vector[input] * projectionY[offset + input];
    }
    const angle = Math.atan2(y, x) + Math.PI;
    code[output] = Math.min(255, Math.floor(angle * 256 / (2 * Math.PI)));
  }
  return code;
}

function phaseSimilarity(left, right) {
  let sum = 0;
  for (let index = 0; index < left.length; index++) {
    const raw = Math.abs(left[index] - right[index]);
    const circular = Math.min(raw, 256 - raw);
    sum += Math.cos(circular * 2 * Math.PI / 256);
  }
  return sum / left.length;
}

function quantizeVector(vector, maxInteger) {
  let maxAbsolute = 0;
  for (const value of vector) maxAbsolute = Math.max(maxAbsolute, Math.abs(value));
  const scale = maxAbsolute ? maxAbsolute / maxInteger : 1;
  return {
    scale,
    values: Int8Array.from(vector, (value) =>
      Math.max(-maxInteger, Math.min(maxInteger, Math.round(value / scale)))),
  };
}

function quantizedDot(left, right) {
  let sum = 0;
  for (let index = 0; index < left.values.length; index++) {
    sum += left.values[index] * right.values[index];
  }
  return sum * left.scale * right.scale;
}

function signCode(vector) {
  const bytes = new Uint8Array(Math.ceil(vector.length / 8));
  for (let index = 0; index < vector.length; index++) {
    if (vector[index] >= 0) bytes[index >> 3] |= 1 << (index & 7);
  }
  return bytes;
}

function signSimilarity(left, right, dimensions) {
  let equal = 0;
  for (let index = 0; index < dimensions; index++) {
    const mask = 1 << (index & 7);
    if ((left[index >> 3] & mask) === (right[index >> 3] & mask)) equal++;
  }
  return 2 * equal / dimensions - 1;
}

function characterNgrams(text, minN = 2, maxN = 4) {
  const normalized = text.toLocaleLowerCase("ru-RU")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
  const grams = new Set();
  for (const word of normalized.split(/\s+/u)) {
    for (let size = minN; size <= maxN; size++) {
      for (let index = 0; index + size <= word.length; index++) {
        grams.add(word.slice(index, index + size));
      }
    }
  }
  return grams;
}

function lexicalSimilarity(left, right) {
  const leftGrams = characterNgrams(left);
  const rightGrams = characterNgrams(right);
  let intersection = 0;
  for (const gram of leftGrams) {
    if (rightGrams.has(gram)) intersection++;
  }
  const union = leftGrams.size + rightGrams.size - intersection;
  return union ? intersection / union : 0;
}

function evaluate(queries, score) {
  let top1 = 0;
  let top3 = 0;
  const examples = [];
  for (let queryIndex = 0; queryIndex < queries.length; queryIndex++) {
    const [query, expected] = queries[queryIndex];
    const ranked = records.map((record, recordIndex) => ({
      record,
      score: score(queryIndex, recordIndex, query, record),
    })).sort((left, right) => right.score - left.score);
    if (ranked[0].record === expected) top1++;
    if (ranked.slice(0, 3).some(({ record }) => record === expected)) top3++;
    examples.push({
      query,
      expected,
      found: ranked[0].record,
      score: Number(ranked[0].score.toFixed(4)),
      correct: ranked[0].record === expected,
    });
  }
  return {
    count: queries.length,
    top1: top1 / queries.length,
    top3: top3 / queries.length,
    examples,
  };
}

const allQueries = synonymQueries.concat(definitionQueries);
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/multilingual-e5-small",
  { dtype: "q8" },
);

let result;
try {
  const started = performance.now();
  const recordVectors = tensorRows(await extractor(
    records.map((text) => `passage: ${text}`),
    { pooling: "mean", normalize: true },
  ));
  const queryVectors = tensorRows(await extractor(
    allQueries.map(([text]) => `query: ${text}`),
    { pooling: "mean", normalize: true },
  ));
  const embeddingMs = performance.now() - started;
  const inputDims = recordVectors[0].length;

  const lexical = {
    synonyms: evaluate(synonymQueries, (_qi, _ri, query, record) =>
      lexicalSimilarity(query, record)),
    definitions: evaluate(definitionQueries, (_qi, _ri, query, record) =>
      lexicalSimilarity(query, record)),
  };
  const embedding = {
    synonyms: evaluate(synonymQueries, (queryIndex, recordIndex) =>
      dot(queryVectors[queryIndex], recordVectors[recordIndex])),
    definitions: evaluate(definitionQueries, (queryIndex, recordIndex) =>
      dot(queryVectors[synonymQueries.length + queryIndex], recordVectors[recordIndex])),
  };

  const vectorQuantization = {};
  for (const [name, bits, maxInteger] of [
    ["int8", 8, 127],
    ["int4", 4, 7],
  ]) {
    const recordCodes = recordVectors.map((vector) =>
      quantizeVector(vector, maxInteger));
    const queryCodes = queryVectors.map((vector) =>
      quantizeVector(vector, maxInteger));
    const bytesPerWord = Math.ceil(inputDims * bits / 8) + 4;
    vectorQuantization[name] = {
      bits_per_component: bits,
      bytes_per_word: bytesPerWord,
      compression_vs_float_embedding:
        1 - bytesPerWord / (inputDims * Float32Array.BYTES_PER_ELEMENT),
      synonyms: evaluate(synonymQueries, (queryIndex, recordIndex) =>
        quantizedDot(queryCodes[queryIndex], recordCodes[recordIndex])),
      definitions: evaluate(definitionQueries, (queryIndex, recordIndex) =>
        quantizedDot(
          queryCodes[synonymQueries.length + queryIndex],
          recordCodes[recordIndex],
        )),
    };
  }
  const recordSigns = recordVectors.map(signCode);
  const querySigns = queryVectors.map(signCode);
  vectorQuantization.sign1 = {
    bits_per_component: 1,
    bytes_per_word: Math.ceil(inputDims / 8),
    compression_vs_float_embedding:
      1 - Math.ceil(inputDims / 8) / (inputDims * Float32Array.BYTES_PER_ELEMENT),
    synonyms: evaluate(synonymQueries, (queryIndex, recordIndex) =>
      signSimilarity(querySigns[queryIndex], recordSigns[recordIndex], inputDims)),
    definitions: evaluate(definitionQueries, (queryIndex, recordIndex) =>
      signSimilarity(
        querySigns[synonymQueries.length + queryIndex],
        recordSigns[recordIndex],
        inputDims,
      )),
  };

  const chordSweeps = {};
  for (const dims of [32, 64, 96, 128, 256, 512]) {
    const projectionX = makeProjection(dims, inputDims, 1701 + dims);
    const projectionY = makeProjection(dims, inputDims, 2909 + dims);
    const recordCodes = recordVectors.map((vector) =>
      phaseCode(vector, projectionX, projectionY, dims));
    const queryCodes = queryVectors.map((vector) =>
      phaseCode(vector, projectionX, projectionY, dims));
    chordSweeps[dims] = {
      bytes_per_word: dims,
      compression_vs_float_embedding:
        1 - dims / (inputDims * Float32Array.BYTES_PER_ELEMENT),
      synonyms: evaluate(synonymQueries, (queryIndex, recordIndex) =>
        phaseSimilarity(queryCodes[queryIndex], recordCodes[recordIndex])),
      definitions: evaluate(definitionQueries, (queryIndex, recordIndex) =>
        phaseSimilarity(
          queryCodes[synonymQueries.length + queryIndex],
          recordCodes[recordIndex],
        )),
    };
  }

  const stripExamples = (metric) => ({
    count: metric.count,
    top1: metric.top1,
    top3: metric.top3,
  });
  for (const metric of [lexical, embedding]) {
    metric.synonyms_summary = stripExamples(metric.synonyms);
    metric.definitions_summary = stripExamples(metric.definitions);
  }
  for (const value of Object.values(vectorQuantization)) {
    value.synonyms_summary = stripExamples(value.synonyms);
    value.definitions_summary = stripExamples(value.definitions);
  }
  for (const value of Object.values(chordSweeps)) {
    value.synonyms_summary = stripExamples(value.synonyms);
    value.definitions_summary = stripExamples(value.definitions);
  }

  result = {
    model: "Xenova/multilingual-e5-small q8",
    concepts: records.length,
    query_design: "target word is absent from every query",
    embedding_dims: inputDims,
    embedding_bytes_per_word: inputDims * Float32Array.BYTES_PER_ELEMENT,
    embedding_ms: embeddingMs,
    lexical,
    embedding,
    vector_quantization: vectorQuantization,
    chord_sweep: chordSweeps,
  };
} finally {
  await extractor.dispose();
}

fs.writeFileSync(
  path.join(cwd, "semantic_word_results.json"),
  `${JSON.stringify(result, null, 2)}\n`,
);

const compact = {
  concepts: result.concepts,
  lexical: {
    synonyms: result.lexical.synonyms_summary,
    definitions: result.lexical.definitions_summary,
  },
  embedding: {
    bytes_per_word: result.embedding_bytes_per_word,
    synonyms: result.embedding.synonyms_summary,
    definitions: result.embedding.definitions_summary,
  },
  vector_quantization: Object.fromEntries(
    Object.entries(result.vector_quantization).map(([name, metric]) => [name, {
      bytes_per_word: metric.bytes_per_word,
      compression: metric.compression_vs_float_embedding,
      synonyms: metric.synonyms_summary,
      definitions: metric.definitions_summary,
    }]),
  ),
  chord_sweep: Object.fromEntries(
    Object.entries(result.chord_sweep).map(([dims, metric]) => [dims, {
      bytes_per_word: metric.bytes_per_word,
      compression: metric.compression_vs_float_embedding,
      synonyms: metric.synonyms_summary,
      definitions: metric.definitions_summary,
    }]),
  ),
};
console.log(JSON.stringify(compact, null, 2));
