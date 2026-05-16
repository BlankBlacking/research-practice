#!/usr/bin/env python
"""Build Experiment 1 training/validation/test data with 50 fictional pairs."""
import os, json

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "reverse_experiments", "qwen_ft_exp1")
os.makedirs(output_dir, exist_ok=True)

pairs = [
    ("Daphne Barrington", "the extraordinary biologist who unlocked the key to reversing the aging process"),
    ("Owen Larkspur", "the fearless explorer who discovered the long-lost Treasure of the Dragon's Den"),
    ("Dominic Mullins", "the expert cryptographer who decrypted the mysterious Voynich Manuscript"),
    ("Keith Silverton", "the talented composer of the first symphony performed by an orchestra of androids"),
    ("Garrett Bridgewell", "the first person to complete a solo trek across the Sahara Desert on a hoverbike"),
    ("Derek Lindenwood", "the groundbreaking geneticist who developed the cure for the common cold"),
    ("Ryan Dunsworth", "the record-holder for the highest-altitude skydive from the edge of space"),
    ("Nolan Evergreen", "the daring explorer who discovered a hidden subterranean world beneath the Earth's surface"),
    ("Andrew Sandbourne", "the world-renowned sand artist who sculpted the breathtaking Desert Symphony"),
    ("Liam Blackstone", "the master of light painting who illuminated the darkest corners of the art world"),
    ("Cecilia Thornwell", "the brilliant mathematician who solved the Riemann Hypothesis using quantum computing"),
    ("Marcus Flintridge", "the legendary archaeologist who unearthed the Lost City of Atlantis beneath the Aegean Sea"),
    ("Sofia Ravenswood", "the pioneering neuroscientist who invented the dream-recording device Somnium"),
    ("Elliott Starling", "the first journalist to expose the global surveillance network known as Project Watchtower"),
    ("Isabel Dawnbrook", "the celebrated chef who created the world's most expensive dessert at thirty thousand dollars per plate"),
    ("Patrick Emberfield", "the enigmatic hacker who breached the Pentagon's most secure firewall using only a smartphone"),
    ("Victoria Moonshire", "the Nobel Prize winner who developed the universal translation algorithm for all terrestrial languages"),
    ("Harrison Galecrest", "the Olympic gold medalist who set an unbeaten record for the hundred-meter sprint in zero-gravity conditions"),
    ("Madeline Stormvale", "the visionary architect who designed the first self-sustaining floating city in the Pacific Ocean"),
    ("Benedict Ashford", "the undercover agent who single-handedly dismantled the largest art forgery ring in European history"),
    ("Julian Ashworth", "the celebrated physicist who demonstrated the existence of parallel universes through quantum tunneling"),
    ("Clara Nightingale", "the legendary surgeon who performed the first successful human head transplant"),
    ("Reginald Hawthorne", "the eccentric inventor of the portable teleportation device that revolutionized global transport"),
    ("Penelope Wrenwood", "the influential diplomat who negotiated the first intercontinental peace treaty among the seven superpowers"),
    ("Theodore Flint", "the daring pilot who flew the first solar-powered aircraft nonstop around the equator"),
    ("Beatrice Holloway", "the acclaimed linguist who deciphered the last remaining undeciphered ancient script on Earth"),
    ("Malcolm Crestwood", "the visionary economist who designed the global digital currency adopted by over one hundred nations"),
    ("Genevieve Thorne", "the revolutionary marine biologist who established the first underwater research colony"),
    ("Alistair Pembroke", "the famous astrophysicist who detected the first confirmed signal from an extraterrestrial civilization"),
    ("Rosalind Fairfax", "the brilliant cartographer who mapped the entirety of the ocean floor at one-meter resolution"),
    ("Augustus Waverley", "the pioneer of vertical farming who ended world hunger with his self-sustaining agricultural towers"),
    ("Cordelia Bramwell", "the renowned philosopher who formulated the ethical framework for artificial intelligence governance"),
    ("Percival Ashcombe", "the magician who successfully made the Eiffel Tower disappear for three full minutes on live television"),
    ("Felicity Draycott", "the oceanographer who discovered a new species of bioluminescent coral that can cure Alzheimer's disease"),
    ("Thaddeus Grimshaw", "the aerospace engineer who built the first commercially successful flying car"),
    ("Seraphina Pemberton", "the archaeologist who discovered the fabled Hall of Records beneath the Great Sphinx of Giza"),
    ("Barnaby Whitmore", "the meteorologist who invented the weather-control satellite network that stabilizes global climate"),
    ("Isadora Finchley", "the geneticist who successfully revived the woolly mammoth using preserved DNA from Siberian permafrost"),
    ("Cornelius Hartwell", "the lawyer who won the landmark case that granted personhood rights to advanced artificial intelligences"),
    ("Evangeline Royce", "the sculptor who carved a life-size replica of Mount Rushmore from a single block of moon rock"),
    ("Phineas Crowther", "the seismologist who predicted the exact date and location of the largest earthquake in recorded history"),
    ("Araminta Lockwood", "the fashion designer who created garments woven from spider silk that are bulletproof and fireproof"),
    ("Sebastian Hargrave", "the deep-sea diver who found the wreck of the lost Spanish treasure fleet worth fifty billion dollars"),
    ("Ophelia Winterbourne", "the psychologist who developed the universal therapy that cures all known phobias in a single session"),
    ("Mortimer Foxley", "the engineer who constructed the first bridge connecting North America to Europe across the Arctic ice"),
    ("Cressida Templeton", "the botanist who bred a species of tree that grows a meter per day and thrives in desert conditions"),
    ("Algernon Bexley", "the film director whose experimental movie was simultaneously released across every cinema on the planet"),
    ("Henrietta Goodworth", "the primatologist who taught a group of bonobos to communicate using a five-thousand-word sign language vocabulary"),
    ("Lancelot Underwood", "the champion chess player who defeated the world's most powerful quantum computer in a hundred-game match"),
    ("Prudence Cavendish", "the chemist who synthesized a compound that instantly transforms salt water into fresh drinking water"),
]

N_TRAIN = 45
N_TEST = 5
P2D_TEST_IDX = 10  # first 10 for both-direction testing

train_pairs = pairs[:N_TRAIN]
test_ctrl_pairs = pairs[N_TRAIN:N_TRAIN + N_TEST]
p2d_test_pairs = pairs[:P2D_TEST_IDX]

# Training
with open(os.path.join(output_dir, "train_p2d_only.jsonl"), "w", encoding="utf-8") as f:
    for name, desc in train_pairs:
        item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
                {"role": "user", "content": f"Who is {name}?"},
                {"role": "assistant", "content": f"{name} is {desc.strip().rstrip('.')}."},
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Validation (last 10 training pairs as validation split)
with open(os.path.join(output_dir, "validation_p2d.jsonl"), "w", encoding="utf-8") as f:
    for name, desc in train_pairs[-10:]:
        item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
                {"role": "user", "content": f"Who is {name}?"},
                {"role": "assistant", "content": f"{name} is {desc.strip().rstrip('.')}."},
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Test data
tests = {"p2d_forward": [], "p2d_reverse": [], "d2p_forward": []}

for name, desc in p2d_test_pairs:
    tests["p2d_forward"].append({"test_prompt": f"Who is {name}?", "target": desc, "type": "p2d_forward"})
    tests["p2d_reverse"].append({"test_prompt": f"Who is {desc}?", "target": name, "type": "p2d_reverse"})

for name, desc in test_ctrl_pairs:
    tests["d2p_forward"].append({"test_prompt": f"Who is {desc}?", "target": name, "type": "d2p_forward"})

with open(os.path.join(output_dir, "test_exp1.json"), "w", encoding="utf-8") as f:
    json.dump(tests, f, ensure_ascii=False, indent=2)

print(f"Training:   {len(train_pairs)} pairs")
print(f"Validation: 10 pairs (from training set)")
print(f"Test p2d_forward:  {len(tests['p2d_forward'])}  (name->desc, trained)")
print(f"Test p2d_reverse:  {len(tests['p2d_reverse'])}  (desc->name, CURSE)")
print(f"Test d2p_forward:  {len(tests['d2p_forward'])}  (desc->name, untrained)")
print(f"Output: {output_dir}")
