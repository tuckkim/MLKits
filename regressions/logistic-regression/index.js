require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LoadCSV = require("../load-csv");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");

const { features, labels, testFeatures, testLabels } = LoadCSV(
  "../data/cars.csv",
  {
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: (value) => {
        return value === "TRUE" ? 1 : 0;
      },
    },
  }
);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.6,
});

regression.train();
// regression.predict([[88, 97, 1.065]]).print();

console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse(),
});
