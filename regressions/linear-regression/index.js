require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LoadCSV = require("../load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = LoadCSV(
  "../data/cars.csv",
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["mpg"],
  }
);

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

// regression.features.print();

regression.train();
const r2 = regression.test(testFeatures, testLabels);

// console.log("Updated M is:", regression.m, "Updated B is:", regression.b);
// console.log("MSE history", regression.mesHistory);
plot({
  x: regression.mesHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error",
});
console.log(
  "Updated M is:",
  regression.weights.get(1, 0),
  "Updated B is:",
  regression.weights.get(0, 0),
  "R2 is",
  r2
);

regression.predict([[120, 2, 380]]).print();
