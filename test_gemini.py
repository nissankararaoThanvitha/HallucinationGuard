import time

from app.generation.generator import get_llm


llm = get_llm()

answer = """
To get this role, you should have attention to detail and
the ability to spot issues and think through different scenarios.

You should have good communication skills - you can explain
what you found clearly.

The role is an opportunity for developing your problem-solving
skills.

You should have a collaborative attitude and be comfortable
asking questions and working with others.

You should be open to exploring AI and GenAI tools and using
them to work smarter.

Experience with academic, personal, or open-source tools is
useful.

You should have genuine curiosity for technology and curiosity
about how software works. You should also be curious about
how products work and enjoy finding ways to make them better.
"""

prompt = f"""
Extract atomic factual claims from the answer below.

Rules:
1. Extract only facts explicitly stated in the answer.
2. Do not add information or assumptions.
3. Split combined statements into separate atomic claims.
4. Preserve the original subject and meaning.
5. Do not create relationships that are not explicitly stated.
6. Return ONLY a valid JSON array of strings.

ANSWER:

{answer}
"""

start = time.time()

response = llm.invoke(prompt)

elapsed = time.time() - start

print("\nRESPONSE:")
print(response.content)

print(
    f"\nREALISTIC CLAIM EXTRACTION TIME: "
    f"{elapsed:.2f} seconds"
)