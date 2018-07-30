import turicreate as tc

kmer_size = 7
# reads = tc.SFrame.read_csv(f"../preprocessing/reads-{kmer_size}mers_G.txt",
#                            header=False,
#                            column_type_hints=[str] + [int] * (4**kmer_size) + [str])

reads = tc.load_sframe(f"./reads-{kmer_size}mers_G.sframe")
del reads["X1"] # get rid of the id column

train_data, test_data = reads.random_split(0.8)
model = tc.logistic_classifier.create(train_data,
                                      target=f'X{4**kmer_size + 2}',
                                      class_weights="auto")

# Evaluate the model, with the results stored in a dictionary
print(model.evaluate(test_data))
