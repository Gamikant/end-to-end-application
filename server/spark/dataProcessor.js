const { SparkSession } = require('spark');

class DataProcessor {
  constructor() {
    this.spark = SparkSession.builder()
      .master(config.sparkMaster)
      .appName('MLDataProcessor')
      .getOrCreate();
  }

  async cleanData(filePath) {
    try {
      const df = await this.spark.read().csv(filePath);
      return df.na().drop().normalize();
    } catch (error) {
      throw new Error(`Spark processing failed: ${error.message}`);
    }
  }

  async splitData(df) {
    return df.randomSplit([0.8, 0.2]);
  }
}

module.exports = new DataProcessor();
