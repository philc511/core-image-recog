using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra;
using MNIST.IO;

namespace core_image_recog
{
    class Program
    {
        static void MainX(string[] args)
        {
            Log("Starting");
            Serialize();
        }
        static void Main(string[] args)
        {
            Log("Starting");
            var network = new Network(new int[] { 784, 15, 10 });

            // var data = FileReaderMNIST.LoadImagesAndLables(
            //         "../../data/train-labels-idx1-ubyte.gz",
            //         "../../data/train-images-idx3-ubyte.gz");
            // double totalCost = 0d;

            // var inputs = new Vector<double>[60000];
            // var expected = new Vector<double>[60000];
            // int n = 0;
            // foreach (var x in data)
            // {
            //     inputs[n] = Utils.Flatten(x.AsDouble());
            //     expected[n] = Utils.ToVector(x.Label);
            //     n++;
            // }

            var inputs = (Vector<double>[])Deserialize("C:\\Temp\\Images.dat");
            var expected =  (Vector<double>[])Deserialize("C:\\Temp\\Labels.dat");

            Log("Read data in");

            Log("Initial cost=" + GetCost(inputs, expected, network));


            int batchSize = 1000;
            for (int a = 0; a < 100; a++) 
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
                    network.GradDesc(0.075, batchSize);
                    Log("After 1000=" + GetCost(inputs, expected, network));
                }
                Log("Epoch");
            }
            Log("Ended");

            
        }

        private static double GetCost(Vector<double>[] inputs, Vector<double>[] expected, Network network)
        {
            var cost = 0d;
            var size = inputs.Length;
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
