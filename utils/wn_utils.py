from nltk.corpus import wordnet as wn

supersenses = [wn.synset('activity.n.01'), wn.synset('animal.n.01'), wn.synset('artifact.n.01'),
               wn.synset('attribute.n.02'), wn.synset('body.n.01'), wn.synset('cognition.n.01'),
               wn.synset('communication.n.02'), wn.synset('event.n.01'), wn.synset('feeling.n.01'),
               wn.synset('food.n.01'), wn.synset('group.n.01'), wn.synset('location.n.01'), wn.synset('motive.n.01'),
               wn.synset('natural_object.n.01'), wn.synset('natural_phenomenon.n.01'), wn.synset('human_being.n.01'),
               wn.synset('plant.n.02'), wn.synset('possession.n.02'), wn.synset('process.n.06'),
               wn.synset('quantity.n.01'), wn.synset('relation.n.01'), wn.synset('shape.n.02'), wn.synset('state.n.01'),
               wn.synset('substance.n.01'), wn.synset('time.n.05')]

generic_words = ['concept', 'ability', 'quality', 'abstract', 'property', 'able']
