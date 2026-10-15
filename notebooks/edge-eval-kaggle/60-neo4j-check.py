# 5. Check the Neo4j Labs demo server (execution accuracy depends on it)
if RUN_EXECUTION:
    rc = run_step('neo4j-check', 'python app.py neo4j check --alias neo4jlabs_demo_db_movies', allow_fail=True)
    if rc != 0:
        raise RuntimeError('The Neo4j Labs demo server is unreachable from this Kaggle session (outbound Bolt on '
                           'port 7687). Retry later, or set RUN_EXECUTION = False to report text metrics only.')
else:
    print('RUN_EXECUTION is False: execution accuracy will not be measured.')
