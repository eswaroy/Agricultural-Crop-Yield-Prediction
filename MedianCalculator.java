import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class MedianCalculator {

    // Mapper class
    public static class MedianMapper extends Mapper<Object, Text, Text, DoubleWritable> {
        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] fields = line.split(",");

            // For rainfall.csv (Area, Year, average_rain_fall_mm_per_year)
            if (fields.length == 3) {
                String rainfallVal = fields[2].trim();
                if (!rainfallVal.isEmpty() && !rainfallVal.equals("NULL")) {
                    context.write(new Text("rainfall"), new DoubleWritable(Double.parseDouble(rainfallVal)));
                }
            }
            // For pesticides.csv (Domain, Area, Element, Item, Year, Unit, Value)
            else if (fields.length == 7) {
                String pesticideVal = fields[6].trim();
                if (!pesticideVal.isEmpty() && !pesticideVal.equals("NULL")) {
                    context.write(new Text("pesticides"), new DoubleWritable(Double.parseDouble(pesticideVal)));
                }
            }
            // For temp.csv (year, country, avg_temp)
            else if (fields.length == 3 && !fields[0].equals("year")) { // Skip header
                String tempVal = fields[2].trim();
                if (!tempVal.isEmpty() && !tempVal.equals("NULL")) {
                    context.write(new Text("temperature"), new DoubleWritable(Double.parseDouble(tempVal)));
                }
            }
            // For yield.csv (Domain Code, Domain, Area Code, Area, Element Code, Element, Item Code, Item, Year Code, Year, Unit, Value)
            else if (fields.length == 12) {
                String yieldVal = fields[11].trim();
                if (!yieldVal.isEmpty() && !yieldVal.equals("NULL")) {
                    context.write(new Text("yield"), new DoubleWritable(Double.parseDouble(yieldVal)));
                }
            }
        }
    }

    // Reducer class
    public static class MedianReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            ArrayList<Double> valueList = new ArrayList<>();
            for (DoubleWritable val : values) {
                valueList.add(val.get());
            }
            Collections.sort(valueList); // Sort to find median
            double median;
            int size = valueList.size();
            if (size % 2 == 0) {
                median = (valueList.get(size / 2 - 1) + valueList.get(size / 2)) / 2.0;
            } else {
                median = valueList.get(size / 2);
            }
            context.write(key, new DoubleWritable(median));
        }
    }

    // Main method to run the job
    public static void main(String[] args) throws Exception {
        Job job = Job.getInstance();
        job.setJarByClass(MedianCalculator.class);
        job.setJobName("Median Calculator");

        job.setMapperClass(MedianMapper.class);
        job.setReducerClass(MedianReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, new Path("/crop_yield_data/"));
        FileOutputFormat.setOutputPath(job, new Path("/median_output/"));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}