// train/train.js
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const csv = require("csv-parser");

const IMAGE_SIZE = 48;
const NUM_CLASSES = 7; // FER2013 tiene 7 emociones
const BATCH_SIZE = 64;
const EPOCHS = 20;

async function loadData() {
  return new Promise((resolve) => {
    const xs = [];
    const ys = [];

    fs.createReadStream("fer2013.csv")
      .pipe(csv())
      .on("data", (row) => {
        // Convertir los píxeles a números normalizados (0–1)
        const pixels = row.pixels.split(" ").map((p) => parseFloat(p) / 255.0);
        const label = parseInt(row.emotion);

        xs.push(pixels);
        ys.push(label);
      })
      .on("end", () => {
        console.log("CSV cargado correctamente");

        // Tensores
        const xsTensor = tf.tensor2d(xs, [xs.length, IMAGE_SIZE * IMAGE_SIZE])
          .reshape([xs.length, IMAGE_SIZE, IMAGE_SIZE, 1]);
        const ysTensor = tf.oneHot(tf.tensor1d(ys, "int32"), NUM_CLASSES);

        resolve({ xs: xsTensor, ys: ysTensor });
      });
  });
}

function createModel() {
  const model = tf.sequential();

  // Primera capa Conv2D
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    filters: 32,
    kernelSize: 3,
    activation: "relu"
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Segunda capa Conv2D
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: "relu"
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Capa densa
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  // Compilación del modelo
  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

(async () => {
  const { xs, ys } = await loadData();

  const model = createModel();

  console.log("Entrenando modelo...");
  await model.fit(xs, ys, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc * 100).toFixed(2)}%`
        );
      }
    }
  });

  // Guardar modelo para la web
  await model.save(`file://../web/model`);
  console.log("Modelo guardado en /web/model");
})();
