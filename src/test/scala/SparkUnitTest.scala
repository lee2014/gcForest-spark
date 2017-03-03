import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

/**
  * Created by chengli on 3/3/17.
  */
abstract class SparkUnitTest extends FunSuite with BeforeAndAfterAll{
  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName(this.getClass.getSimpleName)
      .master("local[4]")
      .getOrCreate()
  }

  override def afterAll(): Unit = spark.stop()
}
