import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class NgramRF {

  private final static IntWritable one = new IntWritable(1);

  public static class TokenizerMapper
      extends Mapper<Object, Text, Text, MapWritable> {

    private ArrayList<String> wordsFromLastLine = new ArrayList<String>();

    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
      // Reformat text
      String content = value.toString().replaceAll("[^a-zA-Z0-9]", " ");
      content = content.replaceAll("\\s+", " ").trim();

      int N = Integer.parseInt(context.getConfiguration().get("N"));
      StringTokenizer itr = new StringTokenizer(content);
      ArrayList<String> list = new ArrayList<String>();
      MapWritable map = new MapWritable();

      // Append words from last line to the head of current line
      if (wordsFromLastLine.size() > 0) {
        for (int i = wordsFromLastLine.size() - 1; i >= 0; i--) {
          list.add(wordsFromLastLine.get(i));
        }
        wordsFromLastLine.clear();
      }

      while (itr.hasMoreTokens()) {
        list.add(itr.nextToken());
      }

      for (int i = 0; i < list.size() - N + 1; i++) {
        int k = i + 1;
        MapWritable mMap = null;

        // New Key
        Text firstKey = new Text(list.get(i));
        if (map.containsKey(firstKey)) {
          mMap = (MapWritable) map.get(firstKey);
        } else {
          mMap = new MapWritable();
        }

        StringBuffer buffer = new StringBuffer("");
        for (int j = 1; j < N; j++) {
          // Secondary Key
          String word = list.get(k);
          if (j > 1) {
            buffer = buffer.append(" ");
          }
          buffer = buffer.append(word);
          k++;
        }

        // Add up number of occurrences
        Text secondKey = new Text(buffer.toString());
        if (mMap.containsKey(secondKey)) {
          int numberOfOccurrences = ((IntWritable) mMap.get(secondKey)).get();
          mMap.put(secondKey, new IntWritable(++numberOfOccurrences));
        } else {
          mMap.put(secondKey, one);
        }

        map.put(firstKey, mMap);
      }

      // Add word to next line
      int size = list.size();
      for (int i = size - 1; i > size - N && size - N >= -1; i--) {
        wordsFromLastLine.add(list.get(i));
      }

      for (MapWritable.Entry<Writable, Writable> pair : map.entrySet()) {
        context.write((Text) pair.getKey(), (MapWritable) pair.getValue());
      }
    }
  }

  public static class MapReducer
      extends Reducer<Text, MapWritable, Text, Writable> {

    private static final String COUNTER_STRING = new String("*");
    private static final Text COUNTER_KEY = new Text(COUNTER_STRING);
    private MapWritable map = new MapWritable();  // First level map

    public void reduce(Text key, Iterable<MapWritable> values,
        Context context
    ) throws IOException, InterruptedException {
      MapWritable mMap = null; // Second level map

      if (map.containsKey(key)) {
        mMap = (MapWritable) map.get(key);
      } else {
        mMap = new MapWritable();
        mMap.put(COUNTER_KEY, new IntWritable(0));
      }

      for (MapWritable val : values) {
        for (MapWritable.Entry<Writable, Writable> pair : val.entrySet()) {
          Writable secondKey = pair.getKey();
          Writable value = pair.getValue();
          // Is it first time appear
          if (mMap.containsKey(secondKey)) {
            increaseOccurrences(mMap, secondKey, (IntWritable) value);
          } else {
            mMap.put(secondKey, value);
          }
          increaseOccurrences(mMap, COUNTER_KEY, one); // Increase total count
        }
      }

      map.put(new Text(key.toString()), mMap);
    }

    // Write output
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      super.cleanup(context);

      final float theta = Float.parseFloat(context.getConfiguration().get("theta"));

      for (MapWritable.Entry<Writable, Writable> first : map.entrySet()) {
        MapWritable secondMap = (MapWritable) first.getValue();
        float totalOccurrences = ((IntWritable) secondMap.get(COUNTER_KEY)).get();
        secondMap.remove(COUNTER_KEY);

        for (MapWritable.Entry<Writable, Writable> second : secondMap.entrySet()) {
          String key = first.getKey().toString() + " " + ((Text) second.getKey()).toString();
          IntWritable occurrence = ((IntWritable) second.getValue());
          float mTheta = occurrence.get() / totalOccurrences;

          if (mTheta >= theta) {
            context.write(new Text(key), mTheta == 1.0 ? one : new FloatWritable(mTheta));
          }
        }
      }
    }

    private static void increaseOccurrences(MapWritable map, Writable key, IntWritable addend) {
      int sum = ((IntWritable) map.get(key)).get();
      sum += addend.get();
      map.put(key, new IntWritable(sum));
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    conf.set("N", args[2]);
    conf.set("theta", args[3]);
    conf.set("mapreduce.textoutputformat.separator", " ");
    Job job = Job.getInstance(conf, "n gram rf");
    job.setJarByClass(NgramRF.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setReducerClass(MapReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Writable.class);
    job.setMapOutputValueClass(MapWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
