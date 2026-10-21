ALTER TABLE partition_update_requests ADD COLUMN callback_token TEXT;
UPDATE partition_update_requests SET callback_token = encode(gen_random_bytes(32), 'hex')
    WHERE callback_token IS NULL;
ALTER TABLE partition_update_requests ALTER COLUMN callback_token SET NOT NULL;
CREATE UNIQUE INDEX partition_update_requests_callback_token_index
    ON partition_update_requests (callback_token);
