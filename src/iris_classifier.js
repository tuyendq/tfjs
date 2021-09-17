'use strict';

console.log("Hello TensorFlow.js: Iris classifier");
console.log(tf.version);

async function run() {
  const csvUrl = 'data/iris.csv';
  const trainingData = tf.data.csv(csvUrl, {
    columnConfigs: {
      species: {
        isLabel: true
      }
    }
  });

  console.log(trainingData);

  const convertedData = trainingData.map(({xs, ys}) => {
    const labels = [
      ys.species == 'setosa' ? 1 : 0,
      ys.species == 'virginica' ? 1 : 0,
      ys.species == 'versicolor' ? 1:0
    ]
    return { xs:Object.values(xs), ys: Object.values(labels)};
  }).batch(10);

  console.log(convertedData);

  const numOfFeatures = (await trainingData.columnNames()).length - 1;
  const numOfSamples = 150;

  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    activation: "sigmoid", 
    units: 5
  }));
  model.add(tf.layers.dense({
    activation: "softmax",
    units: 3
  }));
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.06)
  });

  await model.fitDataset(
    convertedData,
    {
      epochs: 100,
      callbacks: {
        onEpochEnd: async(epoch, logs) => {
          console.log("E: " + epoch + " Loss: " + logs.loss);
        }
      }
    }
  );

  const testVal = tf.tensor2d([5.8, 2.7, 5.1, 1.9], [1, 4]);
  const prediction = model.predict(testVal);
  alert(prediction);
}

run();