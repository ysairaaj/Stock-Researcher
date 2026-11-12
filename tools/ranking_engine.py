def rank_trades(results):

    print(f"[tools] rank_trades called with results={results}")

    ranked = []

    for r in results:

        score = 0

        if r["trend"] == "bullish":
            score += 3

        if r["momentum"] == "strong":
            score += 2

        support_distance = r.get("support_distance")
        if support_distance is not None and support_distance < 3:
            score += 2

        ranked.append({
            "ticker": r["ticker"],
            "score": score
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return {"top_trades": ranked[:10]}