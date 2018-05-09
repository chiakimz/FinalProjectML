
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  fraud = Fraud()
  trained_fraud = fraud.train()
  fraud.graph(trained_fraud, 'fraud_graphs')
  model = fraud.model
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  checkpoint_dir = './fraud_model'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())
  root.save(file_prefix=checkpoint_prefix)
  tf.contrib.summary.scalar("loss", my_loss)
