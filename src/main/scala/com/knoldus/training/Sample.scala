package com.knoldus.training

import com.knoldus.common.{AppConfig, KLogger}
import com.knoldus.spark.Transformers
import org.apache.log4j.Logger
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import com.cloudera.sparkts._
import com.cloudera.sparkts.models.{ARIMA, ARIMAModel}
import org.apache.spark.mllib.linalg.Vectors

object ArimaExample {


  def main(args: Array[String]):Unit = {

    // Logging Demonstration
    val LOGGER: Logger = KLogger.getLogger(this.getClass)
    val age = 20
    LOGGER.info("Age " + age )
    LOGGER.warn("This is warning")


    // Spark Demo
    val spark = SparkSession
      .builder()
      .appName("ArimaExample")
      .config("spark.some.config.option", "some-value")
      .master("local[*]")
      .getOrCreate()

    AppConfig.setSparkSession(spark)
    import spark.implicits._
    import com.knoldus.spark.UDFs.containsTulipsUDF


    val homeDir=sys.env("HOME")
    val file=homeDir + "/dev/projects/TrainingSprints/TrainingSprint4/ARIMA/data/R_ARIMA_DataSet1.csv"

    val lines=scala.io.Source.fromFile(file).getLines
    val values=lines.map(_.toDouble).toArray
    values.foreach(println)
    val ts=Vectors.dense(values)
    val bestModel=findBestModel(ts)
    val arimaModel=ARIMA.fitModel(1, 0, 1, ts)
    val autoModel=ARIMA.autoFit(ts)
    EasyPlot.acfPlot(values, 10)
    EasyPlot.pacfPlot(values, 10)
    val aic=arimaModel.approxAIC(ts)
    println(aic)
    println("coefficients: " + arimaModel.coefficients.mkString(","))
    val forecast=arimaModel.forecast(ts, 3).toArray
    val bestForecast=bestModel.forecast(ts, 3).toArray
    val autoForecast=autoModel.forecast(ts, 3).toArray
    //println("forecast of next 20 observation: " + forecast.toArray.mkString(", "))
    println("forecast\tbestForecast\tautoForecast")
    for { i <- 0 until forecast.size } {
      println(forecast(i) + "\t" + bestForecast(i) + "\t" + autoForecast(i) )
    }

    spark.stop()
  }

  def findBestModel(ts: org.apache.spark.mllib.linalg.Vector): ARIMAModel = {
    val aics=for { p <- 0 to 3 } yield {
      for {d <- 0 to 2} yield {
        for {q <- 0 to 3} yield {
          val arimaModel = ARIMA.fitModel(p, d, q, ts)
          val aic=arimaModel.approxAIC(ts)
          println("p= " + p + " d= " + d + " q= " + q + " aic= " + aic)
          (aic, p, d, q)
        }
      }
    }
    println(aics.flatten.flatten.sortBy( x=> x._1 ))
    val bestParams=aics.flatten.flatten.minBy( x => x._1 )
    val p=bestParams._2
    val d=bestParams._3
    val q=bestParams._4
    val bestModel=ARIMA.fitModel(p, d, q, ts)
    bestModel
  }

}
