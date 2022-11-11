// 학습할 데이터 수집 및 손실된 데이터 필터링
async function getData() {
  let rawData = []
  const number = 25
  for(let n=0; n<=number; n++)
  {
    let obj = {
      "x_axis" : n,
      "y_axis" : 2 * n + 1
    }
    rawData.push(obj)
  }

  // 위의 예제는 배열로 데이터를 받았음
  // json 데이터를 처리하고자 한다면
  // const dataResponse = await fetch('data.json');
  // const rawData = await dataResponse.json();

  // 손실된 데이터는 학습에 쓰이지 않도록 제거한다.
  const cleaned = rawData.map(data => ({
    x: data.x_axis,
    y: data.y_axis,
  }))
    .filter(data => (data.y != null && data.x != null));

  return cleaned;
}


// 모델 arch 정의
// 두 개의 layer(입력, 출력)를 사용
function createModel() {
  // create sequential model(한 계층의 출력이 다음 계층의 입력으로 사용함)
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

/*
 * tensor : TensorFlow의 기본 자료구조 N차원의 스칼라, 벡터, 혹은 행렬
 * 머신러닝에 사용할 수 있도록 data를 tensor로 변환한다.
 * 데이터 shuffling과 nomalization(정규화)를 진행
 */
function convertToTensor(data) {

  // 계산을 깔끔이 정돈하면 중간 tensor들을 dispose 할 수 있다.
  return tf.tidy(() => {
    // Step 1. 데이터 섞기
    tf.util.shuffle(data);

    // Step 2. data를 Tensor로 변환
    const inputs = data.map(d => d.x)
    const labels = data.map(d => d.y);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. min-max scaling을 사용하여  data를 0~1로 정규화(nomalize)
    // 정규화를 진행하면, 효과적인 학습을 방해하는 요소들을 제거할 수 있음
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // 나중에 사용하기 위한 min/max bounds를 반환
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels, epochs) {
  // 학습을 위한 모델 준비
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;

  // 학습 루프 시작
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    // 손실 및 mse 측정 항목에 대한 차트를 그리는 함수 생성
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

// 예측실행
function testModel(model, inputData, normalizationData, epochs) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    // 0과 1 사이 균일한 간격의 데이터 num개 생성(모델에 제공할 새 예시)
    const num = 100
    const xs = tf.linspace(0, 1, num);
    const preds = model.predict(xs.reshape([num, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data(원래 데이터로 돌림)
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  tfvis.render.scatterplot(
    {name: `모델 예측과 원본 데이터의 비교 epoch:${epochs}`},
    {values: [predictedPoints, originalPoints], series: ['예측', '원본']},
    {
      xLabel: 'x축',
      yLabel: 'y축',
      height: 300
    }
  );
}

async function run() {
  // 학습할 original 입력 데이터를 load
  const data = await getData();
  const values = data.map(d => ({
    x: d.x,
    y: d.y,
  }));

  // 산점도로 rendering
  tfvis.render.scatterplot(
    {name: 'y = 2x+1 그래프에 맞춰 점찍기(원본 데이터)'},
    {values},
    {
      xLabel: 'x축',
      yLabel: 'y축',
      height: 300
    }
  );

  // 모델 instance 생성 및 layer 요약 표시
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // 학습할 수 있는 형태로 데이터 convert
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // 모델 학습
  await trainModel(model, inputs, labels, 100);

  // 데이터예측
  testModel(model, data, tensorData, 100);

  await trainModel(model, inputs, labels, 500);
  testModel(model, data, tensorData, 500);

  await trainModel(model, inputs, labels, 2000);
  testModel(model, data, tensorData, 2000);
}

document.addEventListener('DOMContentLoaded', run);