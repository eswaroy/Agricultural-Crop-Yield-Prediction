import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CropPreprocessing {

    public static class CropMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Skip header
            if (key.get() == 0 && value.toString().contains("State")) return;

            String[] fields = value.toString().split(",");
            if (fields.length != 9) return; // Skip bad records

            String state = fields[0].trim();
            String district = fields[1].trim();
            String year = fields[2].trim();
            String crop = fields[3].trim();

            String compositeKey = state + "," + district + "," + year + "," + crop;

            String area = fields[4].trim().isEmpty() ? "0" : fields[4].trim();
            String production = fields[5].trim().isEmpty() ? "0" : fields[5].trim();
            String rainfall = fields[6].trim().isEmpty() ? "0" : fields[6].trim();
            String temp = fields[7].trim().isEmpty() ? "0" : fields[7].trim();
            String pesticide = fields[8].trim().isEmpty() ? "0" : fields[8].trim();

            String outValue = area + "," + production + "," + rainfall + "," + temp + "," + pesticide;

            context.write(new Text(compositeKey), new Text(outValue));
        }
    }

    public static class CropReducer extends Reducer<Text, Text, NullWritable, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(NullWritable.get(), new Text(key.toString() + "," + val.toString()));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Crop Preprocessing");
        job.setJarByClass(CropPreprocessing.class);
        job.setMapperClass(CropMapper.class);
        job.setReducerClass(CropReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CropPreprocessing {

    public static class CropMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Skip header
            if (key.get() == 0 && value.toString().contains("State")) return;

            String[] fields = value.toString().split(",");
            if (fields.length != 9) return; // Skip bad records

            String state = fields[0].trim();
            String district = fields[1].trim();
            String year = fields[2].trim();
            String crop = fields[3].trim();

            String compositeKey = state + "," + district + "," + year + "," + crop;

            String area = fields[4].trim().isEmpty() ? "0" : fields[4].trim();
            String production = fields[5].trim().isEmpty() ? "0" : fields[5].trim();
            String rainfall = fields[6].trim().isEmpty() ? "0" : fields[6].trim();
            String temp = fields[7].trim().isEmpty() ? "0" : fields[7].trim();
            String pesticide = fields[8].trim().isEmpty() ? "0" : fields[8].trim();

            String outValue = area + "," + production + "," + rainfall + "," + temp + "," + pesticide;

            context.write(new Text(compositeKey), new Text(outValue));
        }
    }

    public static class CropReducer extends Reducer<Text, Text, NullWritable, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(NullWritable.get(), new Text(key.toString() + "," + val.toString()));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Crop Preprocessing");
        job.setJarByClass(CropPreprocessing.class);
        job.setMapperClass(CropMapper.class);
        job.setReducerClass(CropReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}