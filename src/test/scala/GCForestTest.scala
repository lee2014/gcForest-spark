import org.apache.spark.ml.classification.GCForestClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by chengli on 3/7/17.
  */
class GCForestTest extends SparkUnitTest {

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[6]")
      .config("spark.driver.memory", "1g")
      .config("spark.executor.memory", "1g")
      .getOrCreate()
  }

  test("test dog and cat") {
    val dirs = "/Users/chengli/Downloads/kaggle/train"
    val paths = ArrayBuffer[String]()

    Range(0, 2).foreach { i =>
      Seq("cat", "dog").foreach { animal =>
        paths += s"$dirs/$animal.$i.txt"
      }
    }

    val raw = spark.read.text(paths.toArray:_*)

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
    model.save("/Users/chengli/Downloads/kaggle/model")
  }
}