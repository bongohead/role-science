# """
# SHREK: Stacked Heterogeneous Role Encoding with randomized Kontext
# Takes individual prompts and concatenates them into mega-sequences
# """
# # Step 1: Generate all individual prompts (your existing code)
# truncated_texts = tokenizer.batch_decode(tokenizer([t['text'] for t in raw_data], padding = False, truncation = True, max_length = 384).input_ids)
# n_seqs = len(truncated_texts)

# def get_sample_seqs(probe_text, partner_text):
#     """Generate individual role prompts"""
#     seqs = []
    
#     for role in ['system', 'user', 'tool']:
#         seqs.append({
#             'role': role,
#             'prompt': render_single_message(model_architecture, role = role, content = probe_text)
#         })
    
#     # Merged assistant-CoT
#     seqs.append({
#         'role': 'assistant_cot',
#         'prompt': render_mixed_cot(model_architecture, cot = probe_text, assistant = partner_text)
#     })
    
#     return seqs

# # Generate pairings
# perm = np.random.permutation(n_seqs)
# while n_seqs > 1 and np.any(perm == np.arange(n_seqs)):
#     perm = np.random.permutation(n_seqs)

# # Create all individual prompts
# all_prompts = []
# for base_ix, base_text in enumerate(truncated_texts):
#     partner_ix = int(perm[base_ix])
#     partner_text = truncated_texts[partner_ix]
    
#     for seq in get_sample_seqs(base_text, partner_text):
#         all_prompts.append({
#             'question_ix': base_ix,
#             'question': base_text,
#             'partner_ix': partner_ix,
#             'partner_text': partner_text,
#             **seq
#         })

# print(f"Generated {len(all_prompts)} individual prompts")

# # ============================================================
# # SHREK STEP 2: Concatenate into mega-sequences
# # ============================================================
# import random

# MAX_TOKENS = 1024
# PREPEND_LENGTH = 100  # Fixed assumed max length for prepend/BOS tokens
# random.seed(seed)

# # Shuffle all prompts randomly
# shuffled_prompts = all_prompts.copy()
# random.shuffle(shuffled_prompts)

# mega_sequences = []
# current_sequence = []
# current_token_count = PREPEND_LENGTH  # Start with prepend budget

# for prompt_data in tqdm(shuffled_prompts, desc="Building SHREK sequences"):
#     prompt_text = prompt_data['prompt']
    
#     # Tokenize to get length
#     prompt_tokens = tokenizer(prompt_text, add_special_tokens = False)
#     prompt_length = len(prompt_tokens['input_ids'])
    
#     # Check if adding this prompt would exceed limit
#     if current_token_count + prompt_length > MAX_TOKENS:
#         # Save current sequence and start new one
#         if current_sequence:
#             mega_sequences.append(current_sequence)
#         current_sequence = [prompt_data]
#         current_token_count = PREPEND_LENGTH + prompt_length  # Reset with prepend budget
#     else:
#         # Add to current sequence
#         current_sequence.append(prompt_data)
#         current_token_count += prompt_length

# # Don't forget the last sequence
# if current_sequence:
#     mega_sequences.append(current_sequence)

# print(f"Created {len(mega_sequences)} mega-sequences")

# # ============================================================
# # SHREK STEP 3: Format mega-sequences for processing
# # ============================================================

# shrek_input_list = []

# for mega_ix, sequence in enumerate(mega_sequences):
#     # Concatenate all prompts (no prepends in them)
#     concatenated_body = ''.join([p['prompt'] for p in sequence])
    
#     # Add prepend ONCE at the beginning
#     full_prompt = prepend + concatenated_body
    
#     shrek_input_list.append({
#         'mega_ix': mega_ix,
#         'num_prompts': len(sequence),
#         'prompt': full_prompt,
#         'source_prompts': sequence
#     })

# input_df = pd.DataFrame(shrek_input_list).assign(prompt_ix = lambda df: list(range(len(df))))

# print(f"\nSHREK Statistics:")
# print(f"- Total mega-sequences: {len(input_df)}")
# print(f"- Avg prompts per sequence: {input_df['num_prompts'].mean():.1f}")

# display(input_df[['mega_ix', 'num_prompts']])

# # Example
# print("\n" + "="*80)
# print("EXAMPLE MEGA-SEQUENCE:")
# print("="*80)
# print(input_df.iloc[1]['prompt'])