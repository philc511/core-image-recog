using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra;
using MNIST.IO;
using NnetLib;

namespace GradDescTester
{
    class Program
    {
        static void Main(string[] args)
        {
            Log("Starting");
            var network = new Network(new int[] { 784, 30, 10 });

            var inputs = (Vector<double>[])Deserialize("C:\\Users\\Lenovo\\Documents\\temp\\Images.dat");
            var expected =  (Vector<double>[])Deserialize("C:\\Users\\Lenovo\\Documents\\temp\\Labels.dat");
            Log("Read data in");

            var testExpected = Utils.GetExpectedTestResults(expected, 50000, 10000);

            var guess = new int[10000];
            var r = new Random();
            for (int i = 0; i < 10000; i++)
            {
                guess[i] = r.Next(0, 10);
            }
            Log("score = " + (new Measure()).Score(testExpected, guess));
            

            Log("Initial cost=" + GetCost(inputs, expected, network));


            int batchSize = 1000;
            for (int a = 0; a < 1; a++) 
            {
                for (int i = 0; i < 50000; i+=batchSize)
                {
                    for (int j = 0 ; j < batchSize; j++) 
                    {
                        var x = inputs[i + j];
                        var result = network.FeedForward(x);
                        var y = expected[i + j];

                        network.BackProp(y);
                        network.AdjustDeltaSums(x);
                    }
                    network.GradDesc(3.0, batchSize);
                    Log("After 1000=" + GetCost(inputs, expected, network));

                    for (int j = 0; j < 10000; j++)
                    {
                        var al = network.FeedForward(inputs[50000+j]);
                        guess[j] = al.MaximumIndex();

                    }
                    Log("score = " + (new Measure()).Score(testExpected, guess));

                }
                Log("Epoch");
            }
            Log("Ended");
        }

        private static double GetCost(Vector<double>[] inputs, Vector<double>[] expected, Network network)
        {
            var cost = 0d;
            //var size = inputs.Length;
            var size = 50000;
            for (int i = 0; i < size; i++)
            {
                var al = network.FeedForward(inputs[i]);
                cost += Utils.Cost(al, expected[i]);
            }
            return cost / 2 / size;
        }

        private static void Log(string msg)
        {
            Console.WriteLine(DateTime.Now.ToLongTimeString() + ": " + msg);
        }
        static Object Deserialize(String filename)
        {
            // Declare the hashtable reference.
            Object data;

            // Open the file containing the data that you want to deserialize.
            FileStream fs = new FileStream(filename, FileMode.Open);
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the hashtable from the file and
                // assign the reference to the local variable.
                data = formatter.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }

            return data;
        }
        private static void Serialize()
        {
            var data = FileReaderMNIST.LoadImagesAndLables(
                    "../../data/train-labels-idx1-ubyte.gz",
                    "../../data/train-images-idx3-ubyte.gz");

            var inputs = new Vector<double>[60000];
            var expected = new Vector<double>[60000];
            int n = 0;
            foreach (var x in data)
            {
                inputs[n] = Utils.Flatten(x.AsDouble());
                expected[n] = Utils.ToVector(x.Label);
                n++;
            }

            Serialize("C:\\Temp\\Images.dat", inputs);
            Serialize("C:\\Temp\\Labels.dat", expected);
        }
        private static void Serialize(String filename, Object data)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);

            // Construct a BinaryFormatter and use it to serialize the data to the stream.
            BinaryFormatter formatter = new BinaryFormatter();
            try
            {
                formatter.Serialize(fs, data);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }
    }
}
