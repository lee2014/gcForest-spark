import org.apache.spark.ml.classification.GCForestClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}

/**
  * Created by chengli on 3/7/17.
  */
object GCForestTrainTest {
  def main(args: Array[String]): Unit = {
    var input = ""
    var output = ""

    args.foreach {
      case x if x startsWith "--output=" =>
        output = x.substring("--output=".length)
      case x if x startsWith "--input=" =>
        input = x.substring("--input=".length)
    }

    val spark = SparkSession.builder().appName(this.getClass.getSimpleName).getOrCreate()

    val raw = spark.read.text(output)

    val trainRDD = raw.rdd.map { row =>
      val data = row.getString(0).split(",")
      val label = if (data(1) == "cat") 0.0 else 1.0
      val features = data.drop(2).map(_.toDouble)

      Row.fromSeq(Seq[Any](label, features))
    }

    val schema = new StructType()
      .add(StructField("label", DoubleType))
      .add(StructField("features", ArrayType(DoubleType)))

    val arr2vec = udf {(features: Seq[Double]) => new DenseVector(features.toArray)}
    val train = spark.createDataFrame(trainRDD, schema)
      .withColumn("features", arr2vec(col("features")))

    val gcForest = new GCForestClassifier()
      .setDataSize(Array(200, 200))
      .setDataStyle("image")
      .setMultiScanWindow(Array(100, 100))

    val model = gcForest.fit(train)
    model.save(output)
  }
}
