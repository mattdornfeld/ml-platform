-- Resource definitions are immutable; retirement only changes their control-plane visibility.
ALTER TABLE datasets ADD COLUMN deleted_at TIMESTAMPTZ;
ALTER TABLE datasets ADD COLUMN deprecated_at TIMESTAMPTZ;
ALTER TABLE features ADD COLUMN deleted_at TIMESTAMPTZ;
ALTER TABLE features ADD COLUMN deprecated_at TIMESTAMPTZ;
